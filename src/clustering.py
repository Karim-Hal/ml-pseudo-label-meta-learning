"""Pseudo-label generation methods used by the LSE benchmark."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import DictionaryLearning
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

SEED = 42


def _safe_n_clusters(n_classes, n_samples):
    n = int(n_classes)
    if n < 2:
        raise ValueError("n_classes must be at least 2")
    if n_samples < n:
        raise ValueError("n_samples must be >= n_classes")
    return n


def pseudo_kmeans(X_scaled, n_classes):
    """k-means pseudo-labels."""
    n_clusters = _safe_n_clusters(n_classes, len(X_scaled))
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    return km.fit_predict(X_scaled)


def pseudo_dbscan(X_scaled, n_classes, min_samples=5):
    """
    DBSCAN pseudo-labels with deterministic eps search and noise reassignment.

    If DBSCAN cannot find at least two non-noise clusters, k-means is used as a
    deterministic fallback so the benchmark records a valid method result.
    """
    if len(X_scaled) <= 2:
        return pseudo_kmeans(X_scaled, n_classes)

    k = min(min_samples, len(X_scaled))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    knn_dists = dists[:, -1]

    labels = None
    for pct in (90, 75, 60, 45, 30):
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
        if len(non_noise_idx) == 0:
            return pseudo_kmeans(X_scaled, n_classes)
        nn1 = NearestNeighbors(n_neighbors=1).fit(X_scaled[~noise_mask])
        _, idx = nn1.kneighbors(X_scaled[noise_mask])
        labels[noise_mask] = labels[non_noise_idx[idx.ravel()]]

    return labels


def pseudo_agglomerative(X_scaled, n_classes):
    """Ward-linkage agglomerative pseudo-labels."""
    n_clusters = _safe_n_clusters(n_classes, len(X_scaled))
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return agg.fit_predict(X_scaled)


def pseudo_gmm(X_scaled, n_classes):
    """Gaussian mixture hard pseudo-labels."""
    n_components = _safe_n_clusters(n_classes, len(X_scaled))
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=SEED,
        max_iter=200,
        reg_covar=1e-5,
    )
    gmm.fit(X_scaled)
    return gmm.predict(X_scaled)


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


def pseudo_autoencoder(
    X_scaled,
    n_classes,
    n_epochs=100,
    batch_size=256,
    max_samples=8000,
):
    """Train an autoencoder, then k-means cluster the latent embeddings."""
    torch.manual_seed(SEED)
    n_classes = _safe_n_clusters(n_classes, len(X_scaled))
    input_dim = X_scaled.shape[1]
    bottleneck_dim = max(n_classes * 2, 8)

    if len(X_scaled) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_scaled), max_samples, replace=False)
        X_fit = X_scaled[idx]
    else:
        X_fit = X_scaled

    tensor_fit = torch.tensor(X_fit, dtype=torch.float32)
    model = _Autoencoder(input_dim, bottleneck_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(tensor_fit))
        for i in range(0, len(tensor_fit), batch_size):
            batch = tensor_fit[perm[i : i + batch_size]]
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, embeddings = model(torch.tensor(X_scaled, dtype=torch.float32))

    return pseudo_kmeans(embeddings.numpy(), n_classes)


def pseudo_dictlearn(X_scaled, n_classes, max_samples=3000):
    """Dictionary-learning pseudo-labels from dominant sparse-code atoms."""
    n_components = _safe_n_clusters(n_classes, len(X_scaled))

    if len(X_scaled) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_scaled), max_samples, replace=False)
        X_fit = X_scaled[idx]
    else:
        X_fit = X_scaled

    dl = DictionaryLearning(
        n_components=n_components,
        max_iter=200,
        random_state=SEED,
        transform_algorithm="lasso_lars",
        n_jobs=1,
    )
    dl.fit(X_fit)
    codes = dl.transform(X_scaled)
    return np.argmax(np.abs(codes), axis=1)


METHODS = {
    "LSE_kmeans": pseudo_kmeans,
    "LSE_dbscan": pseudo_dbscan,
    "LSE_agg": pseudo_agglomerative,
    "LSE_gmm": pseudo_gmm,
    "LSE_autoenc": pseudo_autoencoder,
    "LSE_dictlearn": pseudo_dictlearn,
}
