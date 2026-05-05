"""
Meta-feature extraction.

All extractors receive pre-scaled X_train (float64, StandardScaler applied by notebook 04).
This ensures every option sees the same preprocessed data and no double-scaling occurs.

Five representations:
  Option A — Hand-crafted (~20 label-free features via scipy/sklearn)
  Option B — Autoencoder bottleneck (mean + variance, fixed K=4 → 8 dims)
  Option C — Dictionary Learning sparse codes (mean + variance, fixed K=4 → 8 dims)
  Option D — Distance-based meta-features (Ferrari & de Castro 2015, 19 dims)

Concatenated variants:
  Option A+B — Option A + Option B (8 dims)
  Option A+C — Option A + Option C (8 dims) [main novel claim]
  Option A+D — Option A + Option D (19 dims)
  Option C+D — Option C + Option D (19 dims) [clustering-specific representations]

K is fixed at 4 for Options B and C so all datasets produce the same-dimensional
vector regardless of n_classes. This removes dimensionality as a confound in the
ablation comparison.

Label policy: Option A computes only from X. n_classes is accepted as a scalar
parameter because the user provides k (number of clusters) at deployment — no labels
are required.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import DictionaryLearning, PCA
from sklearn.neighbors import NearestNeighbors
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


# ── Option A: Hand-crafted meta-features (label-free) ─────────────────────────

def extract_optA(X_tr, n_classes):
    """
    Extract ~20 label-free hand-crafted meta-features.

    X_tr must be pre-scaled (StandardScaler applied by the caller).
    n_classes is provided by the user at deployment — no labels are required.
    No y is accepted; all features are derived from X only.
    """
    n, d = X_tr.shape

    feats = {}

    # Size and shape
    feats['n_instances'] = float(n)
    feats['n_features']  = float(d)
    feats['n_classes']   = float(n_classes)

    # Statistical moments (on pre-scaled X)
    feats['skewness_mean']    = float(np.mean(np.abs(skew(X_tr, axis=0))))
    feats['kurtosis_mean']    = float(np.mean(np.abs(kurtosis(X_tr, axis=0))))
    corr = np.corrcoef(X_tr.T)
    mask = ~np.eye(d, dtype=bool)
    feats['mean_abs_pearson'] = float(np.abs(corr[mask]).mean()) if d > 1 else 0.0

    # Clusterability
    feats['hopkins'] = _hopkins(X_tr)

    # Geometry / manifold
    feats['intrinsic_dim_ratio'] = _intrinsic_dim_ratio(X_tr)
    pca1 = PCA(n_components=1, random_state=SEED).fit(X_tr)
    feats['pca_var_pc1'] = float(pca1.explained_variance_ratio_[0])
    pca_top3_var, pca_entropy = _pca_spectrum_stats(X_tr)
    feats['pca_top3_var'] = pca_top3_var
    feats['pca_entropy']  = pca_entropy

    # Distance statistics
    pair_mean, pair_cv, pair_p90 = _pairwise_distance_stats(X_tr)
    feats['pairwise_dist_mean'] = pair_mean
    feats['pairwise_dist_cv']   = pair_cv
    feats['pairwise_dist_p90']  = pair_p90

    k_nn = min(5, max(1, len(X_tr) - 1))
    knn_mean, knn_cv, knn_p90 = _knn_distance_stats(X_tr, k=k_nn)
    feats['knn5_dist_mean'] = knn_mean
    feats['knn5_dist_cv']   = knn_cv
    feats['knn5_dist_p90']  = knn_p90

    # Correlation structure
    feats['feature_sparsity'] = float((np.abs(X_tr) < 0.01).mean())
    high_corr_frac, corr_disp = _correlation_structure(X_tr)
    feats['high_corr_frac']   = high_corr_frac
    feats['corr_dispersion']  = corr_disp

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
    Train an autoencoder on pre-scaled X_tr; return mean + variance of bottleneck activations.

    K is fixed (default 4), giving a 2K=8 dimensional output vector for every dataset.
    X_tr must be pre-scaled (StandardScaler applied by the caller).

    Returns
    -------
    np.ndarray of shape (2*K,): [mean_0..mean_{K-1}, var_0..var_{K-1}]
    """
    torch.manual_seed(SEED)
    X_f = X_tr.astype(np.float32)

    if len(X_f) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_f), max_samples, replace=False)
        X_fit = X_f[idx]
    else:
        X_fit = X_f

    tensor_fit = torch.from_numpy(X_fit)
    model   = _Autoencoder(X_f.shape[1], K)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(tensor_fit))
        for i in range(0, len(tensor_fit), batch_size):
            b = tensor_fit[perm[i : i + batch_size]]
            recon, _ = model(b)
            loss = loss_fn(recon, b)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(torch.from_numpy(X_f))
    z_np = z.numpy()

    return np.concatenate([z_np.mean(0), z_np.var(0)])


# ── Option C: Dictionary Learning sparse codes ─────────────────────────────────

def extract_optC(X_tr, K=FIXED_K, max_samples=3000):
    """
    Fit DictionaryLearning on pre-scaled X_tr; return mean + variance of sparse codes.

    K is fixed (default 4), giving a 2K=8 dimensional output vector.
    X_tr must be pre-scaled (StandardScaler applied by the caller).

    Returns
    -------
    np.ndarray of shape (2*K,): [mean_0..mean_{K-1}, var_0..var_{K-1}]
    """
    if len(X_tr) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), max_samples, replace=False)
        X_fit = X_tr[idx]
    else:
        X_fit = X_tr

    dl = DictionaryLearning(
        n_components=K,
        max_iter=200,
        random_state=SEED,
        transform_algorithm='lasso_lars',
        n_jobs=1,
    )
    dl.fit(X_fit)
    codes = dl.transform(X_tr)

    return np.concatenate([codes.mean(0), codes.var(0)])


# ── Option D: Distance-based meta-features (Ferrari & de Castro 2015) ──────────

def extract_optD(X_tr, max_rows=5000):
    """
    Distance-based meta-features as described in Ferrari & de Castro (2015).
    Returns a 19-element vector: MD1–MD5, MD6–MD15, MD16–MD19.

    X_tr must be pre-scaled (StandardScaler applied by the caller).
    For n > 5000, a random sample of 5000 rows is used to bound runtime.

    Returns
    -------
    np.ndarray of shape (19,)
      MD1–MD5 : mean, variance, std, skewness, kurtosis of min-max-normalised distances
      MD6–MD15: histogram percentages over [0,0.1], (0.1,0.2], ..., (0.9,1.0]
      MD16–MD19: z-score histogram percentages over [0,1), [1,2), [2,3), [3,∞)
    """
    rng = np.random.default_rng(SEED)
    X_sub = (X_tr if len(X_tr) <= max_rows
             else X_tr[rng.choice(len(X_tr), max_rows, replace=False)])

    raw_dists = pdist(X_sub, metric='euclidean')
    if len(raw_dists) == 0:
        return np.zeros(19)

    # Min-max normalize to [0, 1] for MD1–MD15
    d_min, d_max = raw_dists.min(), raw_dists.max()
    norm_dists = (raw_dists - d_min) / (d_max - d_min + 1e-12)

    # MD1–MD5: statistics of normalised distances
    feats = [
        float(np.mean(norm_dists)),   # MD1: mean
        float(np.var(norm_dists)),    # MD2: variance
        float(np.std(norm_dists)),    # MD3: std
        float(skew(norm_dists)),      # MD4: skewness
        float(kurtosis(norm_dists)),  # MD5: kurtosis
    ]

    # MD6–MD15: histogram percentages over 10 equal bins on [0, 1]
    counts, _ = np.histogram(norm_dists, bins=np.linspace(0, 1, 11))
    feats.extend((counts / max(len(norm_dists), 1)).tolist())  # 10 features

    # MD16–MD19: z-score histogram over [0,1), [1,2), [2,3), [3,∞) of raw distances
    z_dists = np.abs((raw_dists - raw_dists.mean()) / (raw_dists.std() + 1e-8))
    z_counts, _ = np.histogram(z_dists, bins=[0, 1, 2, 3, np.inf])
    feats.extend((z_counts / max(len(z_dists), 1)).tolist())  # 4 features

    return np.array(feats, dtype=float)  # 19 total


# ── Option C2: Multi-scale order-invariant DL meta-features ───────────────────

_C2_FEAT_NAMES = [
    'mean', 'var', 'sparsity',
    'l1_p10', 'l1_p50', 'l1_p90',
    'peak', 'recon_mean', 'recon_cv',
]  # 9 features per K value


def extract_optC2(X_tr, K_values=(4, 8, 16), max_samples=3000):
    """
    Multi-scale order-invariant Dictionary Learning meta-features.

    Addresses three limitations of extract_optC:
      1. K=4 alone is too small — uses K=4, 8, 16 (skips if K > n_features)
      2. Per-atom features are order-dependent — all aggregations are global/distributional
      3. Missing reconstruction quality — adds mean and CV of per-row recon error

    X_tr must be pre-scaled (StandardScaler applied by the caller).
    transform_alpha=0.5 gives explicit sparsity control (vs. implicit default).

    Returns
    -------
    dict with 9 * len(K_values) = 27 keys:
      dl2_k{K}_{feat} for feat in [mean, var, sparsity, l1_p10, l1_p50, l1_p90,
                                     peak, recon_mean, recon_cv]
    """
    rng = np.random.default_rng(SEED)
    n, d = X_tr.shape

    if n > max_samples:
        fit_idx = rng.choice(n, max_samples, replace=False)
        X_fit = X_tr[fit_idx]
    else:
        X_fit = X_tr

    feats = {}
    for K in K_values:
        prefix = f'dl2_k{K}'
        if K > d:
            for name in _C2_FEAT_NAMES:
                feats[f'{prefix}_{name}'] = 0.0
            continue
        try:
            dl = DictionaryLearning(
                n_components=K,
                transform_alpha=0.5,
                max_iter=200,
                random_state=SEED,
                transform_algorithm='lasso_lars',
                n_jobs=1,
            )
            dl.fit(X_fit)
            codes = dl.transform(X_tr)
        except Exception:
            for name in _C2_FEAT_NAMES:
                feats[f'{prefix}_{name}'] = float('nan')
            continue

        l1_per_row = np.abs(codes).sum(axis=1)
        reconstruction = codes @ dl.components_
        recon_err = np.linalg.norm(X_tr - reconstruction, axis=1)
        recon_mean = float(recon_err.mean())

        feats[f'{prefix}_mean']       = float(codes.mean())
        feats[f'{prefix}_var']        = float(codes.var())
        feats[f'{prefix}_sparsity']   = float((np.abs(codes) < 0.01).mean())
        feats[f'{prefix}_l1_p10']     = float(np.percentile(l1_per_row, 10))
        feats[f'{prefix}_l1_p50']     = float(np.percentile(l1_per_row, 50))
        feats[f'{prefix}_l1_p90']     = float(np.percentile(l1_per_row, 90))
        feats[f'{prefix}_peak']       = float(np.abs(codes).max(axis=0).mean())
        feats[f'{prefix}_recon_mean'] = recon_mean
        feats[f'{prefix}_recon_cv']   = float(recon_err.std() / (recon_mean + 1e-6))

    return feats


# ── Concatenated variants ──────────────────────────────────────────────────────

def extract_optAB(X_tr, n_classes, K=FIXED_K):
    """Option A + Option B: hand-crafted + autoencoder summary."""
    feats = extract_optA(X_tr, n_classes)
    b_vec = extract_optB(X_tr, K=K)
    for i, v in enumerate(b_vec[:K]):
        feats[f'ae_mean_{i}'] = float(v)
    for i, v in enumerate(b_vec[K:]):
        feats[f'ae_var_{i}'] = float(v)
    return feats


def extract_optAC(X_tr, n_classes, K=FIXED_K):
    """Option A + Option C: hand-crafted + DL sparse code summary.
    Primary novel meta-feature contribution of the project.
    """
    feats = extract_optA(X_tr, n_classes)
    c_vec = extract_optC(X_tr, K=K)
    for i, v in enumerate(c_vec[:K]):
        feats[f'dl_mean_{i}'] = float(v)
    for i, v in enumerate(c_vec[K:]):
        feats[f'dl_var_{i}'] = float(v)
    return feats


def extract_optAD(X_tr, n_classes):
    """Option A + Option D: hand-crafted + distance-based features."""
    feats = extract_optA(X_tr, n_classes)
    d_vec = extract_optD(X_tr)
    for j, v in enumerate(d_vec):
        feats[f'md{j + 1}'] = float(v)
    return feats


def extract_optCD(X_tr, K=FIXED_K):
    """Option C + Option D: DL sparse codes + distance-based features."""
    c_vec = extract_optC(X_tr, K=K)
    d_vec = extract_optD(X_tr)
    rec = {}
    for i, v in enumerate(c_vec[:K]):
        rec[f'dl_mean_{i}'] = float(v)
    for i, v in enumerate(c_vec[K:]):
        rec[f'dl_var_{i}'] = float(v)
    for j, v in enumerate(d_vec):
        rec[f'md{j + 1}'] = float(v)
    return rec


def extract_optAC2(X_tr, n_classes, K_values=(4, 8, 16)):
    """Option A + Option C2: hand-crafted + multi-scale order-invariant DL features."""
    feats = extract_optA(X_tr, n_classes)
    feats.update(extract_optC2(X_tr, K_values=K_values))
    return feats
