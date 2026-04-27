"""
Meta-feature extraction for Phase 3.

Three representations:
  Option A - deployable hand-crafted features (label-safe)
  Option B - autoencoder bottleneck (mean + variance of latent activations)
  Option C - Dictionary Learning sparse codes (mean + variance of codes)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning, PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42


# Option A: hand-crafted, label-safe features

def _hopkins(X, m=None):
    """Hopkins clusterability statistic. H near 1 = clusterable, H near 0.5 = random."""
    n, d = X.shape
    if n < 3:
        return 0.5
    if m is None:
        m = min(max(n // 10, 10), 100, n - 1)
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


def _intrinsic_dim_ratio(X):
    """Number of PCA components to reach 95% variance, divided by n_features."""
    if X.shape[1] == 0:
        return 0.0
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
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return float(entropy / np.log2(len(classes)))


def _imbalance_ratio(y):
    """max_class_count / min_class_count. 1.0 = perfectly balanced."""
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    return float(counts.max() / counts.min()) if len(counts) else 0.0


def _safe_mean_abs(values, default=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(default)
    return float(np.mean(np.abs(values)))


def _resolve_n_classes(y_tr=None, n_classes=None):
    if n_classes is not None:
        return int(n_classes)
    if y_tr is not None:
        return int(len(np.unique(y_tr)))
    return 0


def _kmeans_landmarks(Xs, n_classes):
    """Unsupervised clusterability proxies from a fast k-means pass."""
    n = len(Xs)
    if n_classes < 2 or n < n_classes:
        return {
            "kmeans_est_imbalance_ratio": 0.0,
            "kmeans_largest_cluster_frac": 0.0,
            "kmeans_silhouette": 0.0,
            "kmeans_davies_bouldin": 999.0,
            "kmeans_inertia_per_sample": 0.0,
        }

    km = KMeans(n_clusters=n_classes, random_state=SEED, n_init=10)
    labels = km.fit_predict(Xs)
    _, counts = np.unique(labels, return_counts=True)
    counts = counts[counts > 0]

    feats = {
        "kmeans_est_imbalance_ratio": float(counts.max() / counts.min()),
        "kmeans_largest_cluster_frac": float(counts.max() / counts.sum()),
        "kmeans_inertia_per_sample": float(km.inertia_ / n),
    }

    if 1 < len(np.unique(labels)) < n:
        try:
            feats["kmeans_silhouette"] = float(
                silhouette_score(Xs, labels, sample_size=min(2000, n), random_state=SEED)
            )
        except Exception:
            feats["kmeans_silhouette"] = 0.0
        try:
            feats["kmeans_davies_bouldin"] = float(davies_bouldin_score(Xs, labels))
        except Exception:
            feats["kmeans_davies_bouldin"] = 999.0
    else:
        feats["kmeans_silhouette"] = 0.0
        feats["kmeans_davies_bouldin"] = 999.0

    return feats


def extract_optA(X_tr, y_tr=None, X_te=None, y_te=None, n_cv_folds=5, n_classes=None):
    """
    Extract deployable hand-crafted meta-features from training data.

    This representation is label-safe. y_tr/y_te are accepted for compatibility
    with the old notebooks, but labels are not used as feature signal. If a
    production user knows the expected number of classes, pass n_classes.
    """
    del X_te, y_te, n_cv_folds  # kept only for backward-compatible notebooks

    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    n, d = X_tr.shape
    n_cls = _resolve_n_classes(y_tr=y_tr, n_classes=n_classes)

    feats = {
        "n_instances": float(n),
        "n_features": float(d),
        "n_classes": float(n_cls),
    }

    feats["skewness_mean"] = _safe_mean_abs(skew(Xs, axis=0, nan_policy="omit"))
    feats["kurtosis_mean"] = _safe_mean_abs(kurtosis(Xs, axis=0, nan_policy="omit"))
    if d > 1:
        corr = np.corrcoef(Xs.T)
        mask = ~np.eye(d, dtype=bool)
        feats["mean_abs_pearson"] = _safe_mean_abs(corr[mask])
    else:
        feats["mean_abs_pearson"] = 0.0

    feats["hopkins"] = _hopkins(Xs)
    feats["intrinsic_dim_ratio"] = _intrinsic_dim_ratio(Xs)
    pca = PCA(n_components=1, random_state=SEED).fit(Xs)
    feats["pca_var_pc1"] = float(pca.explained_variance_ratio_[0])
    feats.update(_kmeans_landmarks(Xs, n_cls))

    feats["feature_sparsity"] = float((np.abs(Xs) < 0.01).mean())
    raw_std = X_tr.std(0) + 1e-8
    raw_mean_abs = np.abs(X_tr.mean(0)) + 1e-8
    feats["cv_mean"] = float(np.mean(raw_std / raw_mean_abs))

    return feats


def extract_optA_label_diagnostics(X_tr, y_tr):
    """
    Label-aware diagnostics for offline analysis only.

    Do not feed these columns into the deployable meta-learner; they use labels
    and would leak evaluation information into prediction-time features.
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    feats = {
        "diag_class_entropy": _class_entropy(y_tr),
        "diag_imbalance_ratio": _imbalance_ratio(y_tr),
    }

    classes = np.unique(y_tr)
    if len(classes) < 2:
        feats["diag_inter_intra_ratio"] = 0.0
    else:
        centroids = np.array([Xs[y_tr == c].mean(0) for c in classes])
        dists = [
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(len(classes))
            for j in range(i + 1, len(classes))
        ]
        inter = np.mean(dists)
        intra = np.mean([Xs[y_tr == c].std() + 1e-8 for c in classes])
        feats["diag_inter_intra_ratio"] = float(inter / intra)

    try:
        feats["diag_silhouette_true"] = float(
            silhouette_score(Xs, y_tr, sample_size=min(2000, len(Xs)), random_state=SEED)
        )
    except Exception:
        feats["diag_silhouette_true"] = 0.0
    try:
        feats["diag_davies_bouldin_true"] = float(davies_bouldin_score(Xs, y_tr))
    except Exception:
        feats["diag_davies_bouldin_true"] = 999.0

    return feats


# Option B: autoencoder bottleneck

class _Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        h = max(64, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h),
            nn.ReLU(),
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
    bottleneck_dim = max(int(n_classes) * 2, 8)

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
            b = tensor_fit[perm[i : i + batch_size]]
            recon, _ = model(b)
            loss = loss_fn(recon, b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(torch.from_numpy(Xs))
    z_np = z.numpy()

    mean_vec = z_np.mean(0)
    var_vec = z_np.var(0)
    return np.concatenate([mean_vec, var_vec])


# Option C: Dictionary Learning sparse codes

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
        n_components = max(int(n_classes) * 2, 8)

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
    codes = dl.transform(Xs)

    mean_vec = codes.mean(0)
    var_vec = codes.var(0)
    return np.concatenate([mean_vec, var_vec])
