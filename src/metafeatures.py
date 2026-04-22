"""
Meta-feature extraction for Phase 3.

Three representations:
  Option A — Hand-crafted (~20 features via scipy/sklearn)
  Option B — Autoencoder bottleneck (mean + variance of latent activations)
  Option C — Dictionary Learning sparse codes (mean + variance of codes)
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

SEED = 42


# ── Option A: Hand-crafted features ───────────────────────────────────────────

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


def _class_entropy(y):
    """Normalised class entropy (0 = pure, 1 = uniform)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0.0
    p = counts / counts.sum()
    H = -np.sum(p * np.log2(p + 1e-12))
    return float(H / np.log2(len(classes)))


def _imbalance_ratio(y):
    """max_class_count / min_class_count. 1.0 = perfectly balanced."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    return float(counts.max() / counts.min())


def extract_optA(X_tr, y_tr, X_te, y_te, n_cv_folds=5):
    """
    Extract ~20 hand-crafted meta-features from training data.

    Parameters
    ----------
    X_tr, y_tr : training features (raw, unscaled) and labels
    X_te, y_te : test features and labels (unused here; kept for API symmetry)
    n_cv_folds : folds for landmarker CV

    Returns
    -------
    dict mapping feature name → float
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    n, d = X_tr.shape
    classes = np.unique(y_tr)
    n_cls = len(classes)

    feats = {}

    # --- General ---
    feats["n_instances"] = float(n)
    feats["n_features"] = float(d)
    feats["n_classes"] = float(n_cls)

    # --- Statistical ---
    feats["skewness_mean"] = float(np.mean(np.abs(skew(Xs, axis=0))))
    feats["kurtosis_mean"] = float(np.mean(np.abs(kurtosis(Xs, axis=0))))
    corr = np.corrcoef(Xs.T)
    mask = ~np.eye(d, dtype=bool)
    feats["mean_abs_pearson"] = float(np.abs(corr[mask]).mean()) if d > 1 else 0.0

    # --- Class distribution ---
    feats["class_entropy"] = _class_entropy(y_tr)
    feats["imbalance_ratio"] = _imbalance_ratio(y_tr)

    # --- Clusterability ---
    feats["hopkins"] = _hopkins(Xs)

    # --- Geometry ---
    feats["intrinsic_dim_ratio"] = _intrinsic_dim_ratio(Xs)
    pca = PCA(n_components=1, random_state=SEED).fit(Xs)
    feats["pca_var_pc1"] = float(pca.explained_variance_ratio_[0])
    feats["inter_intra_ratio"] = _inter_intra_ratio(Xs, y_tr)

    # --- Cluster quality with true labels (separability proxies) ---
    try:
        feats["silhouette_true"] = float(silhouette_score(Xs, y_tr, sample_size=2000, random_state=SEED))
    except Exception:
        feats["silhouette_true"] = 0.0
    try:
        feats["davies_bouldin_true"] = float(davies_bouldin_score(Xs, y_tr))
    except Exception:
        feats["davies_bouldin_true"] = 999.0

    # --- Landmarkers ---
    cv = StratifiedKFold(n_splits=min(n_cv_folds, n_cls), shuffle=True, random_state=SEED)

    try:
        knn = KNeighborsClassifier(n_neighbors=1)
        feats["knn1_accuracy"] = float(cross_val_score(knn, Xs, y_tr, cv=cv, scoring="accuracy").mean())
    except Exception:
        feats["knn1_accuracy"] = 0.0

    try:
        stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
        feats["decision_stump_accuracy"] = float(
            cross_val_score(stump, Xs, y_tr, cv=cv, scoring="accuracy").mean()
        )
    except Exception:
        feats["decision_stump_accuracy"] = 0.0

    # --- Sparsity ---
    feats["feature_sparsity"] = float((np.abs(Xs) < 0.01).mean())

    # --- Coefficient of variation (mean across features) ---
    raw_std = X_tr.std(0) + 1e-8
    raw_mean_abs = np.abs(X_tr.mean(0)) + 1e-8
    feats["cv_mean"] = float(np.mean(raw_std / raw_mean_abs))

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


def extract_optB(X_tr, n_classes, n_epochs=100, batch_size=256, max_samples=8000):
    """
    Train an autoencoder on X_tr; return mean + variance of bottleneck activations.

    Returns
    -------
    np.ndarray of shape (2 * bottleneck_dim,): [mean..., var...]
    """
    torch.manual_seed(SEED)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr).astype(np.float32)

    input_dim = Xs.shape[1]
    bottleneck_dim = max(n_classes * 2, 8)

    if len(Xs) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(Xs), max_samples, replace=False)
        X_fit = Xs[idx]
    else:
        X_fit = Xs

    tensor_fit = torch.from_numpy(X_fit)
    model = _Autoencoder(input_dim, bottleneck_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(tensor_fit))
        for i in range(0, len(tensor_fit), batch_size):
            b = tensor_fit[perm[i: i + batch_size]]
            recon, _ = model(b)
            loss = loss_fn(recon, b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(torch.from_numpy(Xs))
    z_np = z.numpy()  # (n_train, bottleneck_dim)

    mean_vec = z_np.mean(0)
    var_vec = z_np.var(0)
    return np.concatenate([mean_vec, var_vec])


# ── Option C: Dictionary Learning sparse codes ─────────────────────────────────

def extract_optC(X_tr, n_classes, n_components=None, max_samples=3000):
    """
    Fit DictionaryLearning on X_tr; return mean + variance of sparse codes.

    Returns
    -------
    np.ndarray of shape (2 * n_components,): [mean..., var...]
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)

    if n_components is None:
        n_components = max(n_classes * 2, 8)

    if len(Xs) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(Xs), max_samples, replace=False)
        X_fit = Xs[idx]
    else:
        X_fit = Xs

    dl = DictionaryLearning(
        n_components=n_components,
        max_iter=200,
        random_state=SEED,
        transform_algorithm="lasso_lars",
        n_jobs=1,
    )
    dl.fit(X_fit)
    codes = dl.transform(Xs)  # (n_train, n_components)

    mean_vec = codes.mean(0)
    var_vec = codes.var(0)
    return np.concatenate([mean_vec, var_vec])
