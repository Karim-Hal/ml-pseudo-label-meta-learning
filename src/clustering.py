"""Six pseudo-label generation methods for tabular classification.

Soft-label policy (Option H — hard labels for all methods)
----------------------------------------------------------
GMM and Dictionary Learning produce soft outputs (posteriors / sparse codes).
We use Option H: argmax to obtain hard cluster assignments, then pass through
the standard Hungarian → RF pipeline identically to the hard-label methods.

This choice makes the comparison between methods clean: every method feeds
the same downstream pipeline. Option S (soft targets with per-row weights)
is a future ablation; it requires modifying the RF training step.

DBSCAN edge cases
-----------------
- Noise points (label -1): reassigned to their nearest non-noise cluster
  via Euclidean distance in feature space before Hungarian alignment.
- More clusters than classes: Hungarian pads with zero columns; unmatched
  clusters map to the majority class (handled in hungarian.py).
- Fewer clusters than classes: fine — Hungarian leaves some classes unmapped;
  LSE will reflect the quality loss naturally.
- Fewer than 2 non-noise clusters: falls back to k-means pseudo-labels."""

import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import DictionaryLearning
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn

SEED = 42


# ── 1. k-means ─────────────────────────────────────────────────────────────────

def pseudo_kmeans(X_scaled, n_classes):
    km = KMeans(n_clusters=n_classes, random_state=SEED, n_init=10)
    return km.fit_predict(X_scaled)


# ── 2. DBSCAN (auto-tune eps, noise reassignment) ──────────────────────────────

def pseudo_dbscan(X_scaled, n_classes):
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    knn_dists = dists[:, -1]

    labels = None
    for pct in [90, 75, 60, 45, 30]:
        eps = float(np.percentile(knn_dists, pct))
        if eps <= 0:
            continue
        db = DBSCAN(eps=eps, min_samples=k).fit(X_scaled)
        n_clusters = len(set(db.labels_) - {-1})
        if n_clusters >= 2:
            labels = db.labels_.copy()
            break

    if labels is None or len(set(labels) - {-1}) < 2:
        return pseudo_kmeans(X_scaled, n_classes)

    noise_mask = labels == -1
    if noise_mask.any():
        non_noise_idx = np.where(~noise_mask)[0]
        nn1 = NearestNeighbors(n_neighbors=1).fit(X_scaled[~noise_mask])
        _, idx = nn1.kneighbors(X_scaled[noise_mask])
        labels[noise_mask] = labels[non_noise_idx[idx.ravel()]]

    return labels


# ── 3. Agglomerative (Ward linkage) ────────────────────────────────────────────

def pseudo_agglomerative(X_scaled, n_classes):
    agg = AgglomerativeClustering(n_clusters=n_classes, linkage='ward')
    return agg.fit_predict(X_scaled)


# ── 4. GMM ─────────────────────────────────────────────────────────────────────

def pseudo_gmm(X_scaled, n_classes):
    gmm = GaussianMixture(
        n_components=n_classes,
        random_state=SEED,
        max_iter=200,
        reg_covar=1e-5,
    )
    gmm.fit(X_scaled)
    return gmm.predict(X_scaled)


# ── 5. Autoencoder + k-means ───────────────────────────────────────────────────

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


def pseudo_autoencoder(X_scaled, n_classes, n_epochs=100, batch_size=256):
    torch.manual_seed(SEED)
    input_dim      = X_scaled.shape[1]
    bottleneck_dim = max(n_classes * 2, 8)

    MAX_AE_SAMPLES = 8000
    if len(X_scaled) > MAX_AE_SAMPLES:
        rng   = np.random.default_rng(SEED)
        idx   = rng.choice(len(X_scaled), MAX_AE_SAMPLES, replace=False)
        X_fit = X_scaled[idx]
    else:
        X_fit = X_scaled

    tensor_fit = torch.tensor(X_fit, dtype=torch.float32)
    model      = _Autoencoder(input_dim, bottleneck_dim)
    optim      = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn    = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(tensor_fit))
        for i in range(0, len(tensor_fit), batch_size):
            batch = tensor_fit[perm[i:i + batch_size]]
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        _, embeddings = model(torch.tensor(X_scaled, dtype=torch.float32))
    embeddings = embeddings.numpy()

    return pseudo_kmeans(embeddings, n_classes)


# ── 6. Dictionary Learning ─────────────────────────────────────────────────────

def pseudo_dictlearn(X_scaled, n_classes):
    n_components = max(n_classes, 2)

    MAX_DL_SAMPLES = 3000
    if len(X_scaled) > MAX_DL_SAMPLES:
        rng   = np.random.default_rng(SEED)
        idx   = rng.choice(len(X_scaled), MAX_DL_SAMPLES, replace=False)
        X_fit = X_scaled[idx]
    else:
        X_fit = X_scaled

    dl = DictionaryLearning(
        n_components=n_components,
        max_iter=200,
        random_state=SEED,
        transform_algorithm='lasso_lars',
        n_jobs=1,
    )
    dl.fit(X_fit)

    codes = dl.transform(X_scaled)
    return np.argmax(np.abs(codes), axis=1)
