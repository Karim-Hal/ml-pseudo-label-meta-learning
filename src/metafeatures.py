"""
Meta-feature extraction.

Three representations:
  Option A — Hand-crafted (~30 features via scipy/sklearn)
  Option B — Autoencoder bottleneck (mean + variance, fixed K=4 → 8 dims)
  Option C — Dictionary Learning sparse codes (mean + variance, fixed K=4 → 8 dims)

Concatenated variants:
  Option A+B — Option A features concatenated with Option B (8 dims)
  Option A+C — Option A features concatenated with Option C (8 dims) [main novel claim]

K is fixed at 4 for Options B and C so all datasets produce the same-dimensional
vector regardless of n_classes.  This removes "more dimensions wins" as a confound
in the ablation comparison.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import DictionaryLearning, PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist

SEED = 42
FIXED_K = 4  # bottleneck / n_components for Options B and C


# ── Option A helpers ───────────────────────────────────────────────────────────

def _hopkins(X, m=None):
    """Hopkins clusterability statistic. H≈1 = clusterable, H≈0.5 = random."""
    n, d = X.shape
    if m is None:
        m = min(max(n // 10, 10), 100)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, m, replace=False)
    X_s = X[idx]
    rest = np.delete(X, idx, axis=0)
    if len(rest) == 0:
        return 0.5
    nbrs = NearestNeighbors(n_neighbors=1).fit(rest)
    u = nbrs.kneighbors(X_s)[0].ravel() ** d
    mins, maxs = X.min(0), X.max(0)
    X_rand = rng.uniform(mins, maxs, size=(m, d))
    w = nbrs.kneighbors(X_rand)[0].ravel() ** d
    denom = u.sum() + w.sum()
    return float(w.sum() / denom) if denom > 0 else 0.5


def _inter_intra_ratio(X, y):
    """Mean inter-class centroid distance / mean intra-class spread."""
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    centroids = np.array([X[y == c].mean(0) for c in classes])
    dists = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(classes)) for j in range(i + 1, len(classes))
    ]
    inter = np.mean(dists)
    intra = np.mean([X[y == c].std() + 1e-8 for c in classes])
    return float(inter / intra)


def _intrinsic_dim_ratio(X):
    """Number of PCA components to reach 95% variance, divided by n_features."""
    pca = PCA(random_state=SEED)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, 0.95)) + 1
    return float(k / X.shape[1])


def _sample_rows(X, max_rows):
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx]


def _pairwise_distance_stats(X, max_rows=1200):
    X_sub = _sample_rows(X, max_rows=max_rows)
    if len(X_sub) < 3:
        return 0.0, 0.0, 0.0
    dists = pdist(X_sub, metric='euclidean')
    mean_d = float(np.mean(dists))
    std_d = float(np.std(dists))
    cv_d = std_d / (mean_d + 1e-8)
    p90_d = float(np.percentile(dists, 90))
    return mean_d, cv_d, p90_d


def _knn_distance_stats(X, k=5):
    if len(X) <= k:
        return 0.0, 0.0, 0.0
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    kth = dists[:, -1]
    mean_k = float(np.mean(kth))
    cv_k = float(np.std(kth) / (mean_k + 1e-8))
    p90_k = float(np.percentile(kth, 90))
    return mean_k, cv_k, p90_k


def _pca_spectrum_stats(X):
    pca = PCA(random_state=SEED).fit(X)
    var = np.clip(pca.explained_variance_ratio_, 1e-12, None)
    top3 = float(var[: min(3, len(var))].sum())
    if len(var) <= 1:
        return top3, 0.0
    entropy = float(-(var * np.log(var)).sum() / np.log(len(var)))
    return top3, entropy


def _correlation_structure(X):
    d = X.shape[1]
    if d < 2:
        return 0.0, 0.0
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    mask = ~np.eye(d, dtype=bool)
    abs_corr = np.abs(corr[mask])
    high_corr_frac = float((abs_corr > 0.8).mean())
    corr_dispersion = float(abs_corr.std())
    return high_corr_frac, corr_dispersion


def _class_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0.0
    p = counts / counts.sum()
    H = -np.sum(p * np.log2(p + 1e-12))
    return float(H / np.log2(len(classes)))


def _imbalance_ratio(y):
    counts = np.bincount(y)
    counts = counts[counts > 0]
    return float(counts.max() / counts.min())


# ── Option A: Hand-crafted meta-features ──────────────────────────────────────

def extract_optA(X_tr, y_tr, X_te=None, y_te=None, n_cv_folds=5):
    """
    Extract ~30 hand-crafted meta-features from training data.

    X_te, y_te are accepted for API symmetry but not used internally.
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    n, d = X_tr.shape
    n_cls = len(np.unique(y_tr))

    feats = {}

    # General
    feats['n_instances'] = float(n)
    feats['n_features']  = float(d)
    feats['n_classes']   = float(n_cls)

    # Statistical
    feats['skewness_mean']   = float(np.mean(np.abs(skew(Xs, axis=0))))
    feats['kurtosis_mean']   = float(np.mean(np.abs(kurtosis(Xs, axis=0))))
    corr = np.corrcoef(Xs.T)
    mask = ~np.eye(d, dtype=bool)
    feats['mean_abs_pearson'] = float(np.abs(corr[mask]).mean()) if d > 1 else 0.0

    # Class distribution
    feats['class_entropy']   = _class_entropy(y_tr)
    feats['imbalance_ratio'] = _imbalance_ratio(y_tr)

    # Clusterability
    feats['hopkins'] = _hopkins(Xs)

    # Geometry
    feats['intrinsic_dim_ratio'] = _intrinsic_dim_ratio(Xs)
    pca1 = PCA(n_components=1, random_state=SEED).fit(Xs)
    feats['pca_var_pc1'] = float(pca1.explained_variance_ratio_[0])
    pca_top3_var, pca_entropy = _pca_spectrum_stats(Xs)
    feats['pca_top3_var']  = pca_top3_var
    feats['pca_entropy']   = pca_entropy
    feats['inter_intra_ratio'] = _inter_intra_ratio(Xs, y_tr)

    # Unsupervised density / geometry
    pair_mean, pair_cv, pair_p90 = _pairwise_distance_stats(Xs)
    feats['pairwise_dist_mean'] = pair_mean
    feats['pairwise_dist_cv']   = pair_cv
    feats['pairwise_dist_p90']  = pair_p90

    k_nn = min(5, max(1, len(Xs) - 1))
    knn_mean, knn_cv, knn_p90 = _knn_distance_stats(Xs, k=k_nn)
    feats['knn5_dist_mean'] = knn_mean
    feats['knn5_dist_cv']   = knn_cv
    feats['knn5_dist_p90']  = knn_p90

    # Cluster quality with true labels (separability proxies)
    try:
        feats['silhouette_true'] = float(
            silhouette_score(Xs, y_tr, sample_size=2000, random_state=SEED))
    except Exception:
        feats['silhouette_true'] = 0.0
    try:
        feats['davies_bouldin_true'] = float(davies_bouldin_score(Xs, y_tr))
    except Exception:
        feats['davies_bouldin_true'] = 999.0

    # Landmarkers
    n_splits = min(n_cv_folds, n_cls, n)
    if n_splits < 2:
        feats['knn1_accuracy']           = 0.0
        feats['decision_stump_accuracy'] = 0.0
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        try:
            knn = KNeighborsClassifier(n_neighbors=1)
            feats['knn1_accuracy'] = float(
                cross_val_score(knn, Xs, y_tr, cv=cv, scoring='accuracy').mean())
        except Exception:
            feats['knn1_accuracy'] = 0.0
        try:
            stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
            feats['decision_stump_accuracy'] = float(
                cross_val_score(stump, Xs, y_tr, cv=cv, scoring='accuracy').mean())
        except Exception:
            feats['decision_stump_accuracy'] = 0.0

    # Sparsity / correlation structure
    feats['feature_sparsity']     = float((np.abs(Xs) < 0.01).mean())
    high_corr_frac, corr_disp    = _correlation_structure(Xs)
    feats['high_corr_frac']       = high_corr_frac
    feats['corr_dispersion']      = corr_disp
    feats['feature_std_dispersion'] = float(np.std(X_tr.std(axis=0)))
    feats['zero_variance_frac']   = float((X_tr.std(axis=0) < 1e-8).mean())

    raw_std      = X_tr.std(0) + 1e-8
    raw_mean_abs = np.abs(X_tr.mean(0)) + 1e-8
    cv_per_feat  = np.clip(raw_std / raw_mean_abs, 0, 100)
    feats['cv_mean'] = float(np.mean(cv_per_feat))

    return feats


# ── Option B: Autoencoder bottleneck ──────────────────────────────────────────

class _Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        h = max(64, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(),
            nn.Linear(h, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h), nn.ReLU(),
            nn.Linear(h, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def extract_optB(X_tr, K=FIXED_K, n_epochs=100, batch_size=256, max_samples=8000):
    """
    Train an autoencoder on X_tr; return mean + variance of bottleneck activations.

    K is fixed (default 4), giving a 2K=8 dimensional output vector for every dataset.
    This ensures all datasets produce the same-length meta-feature vector regardless
    of n_classes, removing dimensionality as a confound in ablation comparisons.

    Returns
    -------
    np.ndarray of shape (2*K,): [mean_0..mean_{K-1}, var_0..var_{K-1}]
    """
    torch.manual_seed(SEED)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr).astype(np.float32)

    if len(Xs) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(Xs), max_samples, replace=False)
        X_fit = Xs[idx]
    else:
        X_fit = Xs

    tensor_fit = torch.from_numpy(X_fit)
    model  = _Autoencoder(Xs.shape[1], K)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(tensor_fit))
        for i in range(0, len(tensor_fit), batch_size):
            b = tensor_fit[perm[i: i + batch_size]]
            recon, _ = model(b)
            loss = loss_fn(recon, b)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(torch.from_numpy(Xs))
    z_np = z.numpy()

    return np.concatenate([z_np.mean(0), z_np.var(0)])


# ── Option C: Dictionary Learning sparse codes ─────────────────────────────────

def extract_optC(X_tr, K=FIXED_K, max_samples=3000):
    """
    Fit DictionaryLearning on X_tr; return mean + variance of sparse codes.

    K is fixed (default 4), giving a 2K=8 dimensional output vector for every dataset.

    Returns
    -------
    np.ndarray of shape (2*K,): [mean_0..mean_{K-1}, var_0..var_{K-1}]
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)

    if len(Xs) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(Xs), max_samples, replace=False)
        X_fit = Xs[idx]
    else:
        X_fit = Xs

    dl = DictionaryLearning(
        n_components=K,
        max_iter=200,
        random_state=SEED,
        transform_algorithm='lasso_lars',
        n_jobs=1,
    )
    dl.fit(X_fit)
    codes = dl.transform(Xs)

    return np.concatenate([codes.mean(0), codes.var(0)])


# ── Concatenated variants ──────────────────────────────────────────────────────

def extract_optAB(X_tr, y_tr, X_te=None, y_te=None, K=FIXED_K):
    """Option A + Option B: hand-crafted features concatenated with autoencoder summary."""
    feats = extract_optA(X_tr, y_tr, X_te, y_te)
    b_vec = extract_optB(X_tr, K=K)
    for i, v in enumerate(b_vec[:K]):
        feats[f'ae_mean_{i}'] = float(v)
    for i, v in enumerate(b_vec[K:]):
        feats[f'ae_var_{i}'] = float(v)
    return feats


def extract_optAC(X_tr, y_tr, X_te=None, y_te=None, K=FIXED_K):
    """Option A + Option C: hand-crafted features concatenated with DL sparse code summary.
    This is the primary novel meta-feature contribution of the project.
    """
    feats = extract_optA(X_tr, y_tr, X_te, y_te)
    c_vec = extract_optC(X_tr, K=K)
    for i, v in enumerate(c_vec[:K]):
        feats[f'dl_mean_{i}'] = float(v)
    for i, v in enumerate(c_vec[K:]):
        feats[f'dl_var_{i}'] = float(v)
    return feats
