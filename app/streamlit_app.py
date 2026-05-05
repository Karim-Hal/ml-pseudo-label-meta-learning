"""
Pseudo-Label Advisor — label-free workflow
Upload unlabeled CSV → specify classes → get method recommendation
→ run clustering → Claude maps clusters to class names → download labeled dataset
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
MODELS_DIR = os.path.join(ROOT, "outputs", "models")

from metafeatures import (
    _hopkins, _intrinsic_dim_ratio, _pairwise_distance_stats,
    _knn_distance_stats, _pca_spectrum_stats, _correlation_structure,
)
from clustering import (
    pseudo_kmeans, pseudo_dbscan, pseudo_agglomerative,
    pseudo_gmm, pseudo_autoencoder, pseudo_dictlearn,
)

METHOD_DISPLAY = {
    "kmeans":    "K-Means",
    "dbscan":    "DBSCAN",
    "agg":       "Agglomerative (Ward)",
    "gmm":       "Gaussian Mixture Model",
    "autoenc":   "Autoencoder + K-Means",
    "dictlearn": "Dictionary Learning",
}
METHOD_FNS = {
    "kmeans":    pseudo_kmeans,
    "dbscan":    pseudo_dbscan,
    "agg":       pseudo_agglomerative,
    "gmm":       pseudo_gmm,
    "autoenc":   pseudo_autoencoder,
    "dictlearn": pseudo_dictlearn,
}
METHOD_COLORS = {
    "kmeans":    "#4C72B0",
    "dbscan":    "#DD8452",
    "agg":       "#55A868",
    "gmm":       "#C44E52",
    "autoenc":   "#8172B2",
    "dictlearn": "#937860",
}
CONFIDENCE_FLOOR = 0.60
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pseudo-Label Advisor",
    page_icon="🔬",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf_path = os.path.join(MODELS_DIR, "meta_clf_optA.pkl")
    reg_path = os.path.join(MODELS_DIR, "meta_reg_optA.pkl")
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        return None, None
    with open(clf_path, "rb") as f:
        clf_bundle = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg_bundle = pickle.load(f)
    return clf_bundle, reg_bundle


# ─────────────────────────────────────────────────────────────────────────────
# Label-free meta-feature extraction
# Features that require true labels (class_entropy, imbalance_ratio,
# inter_intra_ratio, silhouette_true, davies_bouldin_true, knn1_accuracy,
# decision_stump_accuracy) are set to NaN — the pipeline's SimpleImputer
# fills them with the training medians at inference time.
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_unlabeled(X, n_classes):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    n, d = X.shape

    feats = {}

    feats["n_instances"] = float(n)
    feats["n_features"]  = float(d)
    feats["n_classes"]   = float(n_classes)

    feats["skewness_mean"]   = float(np.mean(np.abs(skew(Xs, axis=0))))
    feats["kurtosis_mean"]   = float(np.mean(np.abs(kurtosis(Xs, axis=0))))
    corr = np.corrcoef(Xs.T)
    mask = ~np.eye(d, dtype=bool)
    feats["mean_abs_pearson"] = float(np.abs(corr[mask]).mean()) if d > 1 else 0.0

    # Label-dependent — imputed from training medians at inference
    feats["class_entropy"]          = np.nan
    feats["imbalance_ratio"]        = np.nan
    feats["inter_intra_ratio"]      = np.nan
    feats["silhouette_true"]        = np.nan
    feats["davies_bouldin_true"]    = np.nan
    feats["knn1_accuracy"]          = np.nan
    feats["decision_stump_accuracy"] = np.nan

    feats["hopkins"]              = _hopkins(Xs)
    feats["intrinsic_dim_ratio"]  = _intrinsic_dim_ratio(Xs)

    pca1 = PCA(n_components=1, random_state=SEED).fit(Xs)
    feats["pca_var_pc1"] = float(pca1.explained_variance_ratio_[0])
    pca_top3, pca_ent = _pca_spectrum_stats(Xs)
    feats["pca_top3_var"] = pca_top3
    feats["pca_entropy"]  = pca_ent

    pair_mean, pair_cv, pair_p90 = _pairwise_distance_stats(Xs)
    feats["pairwise_dist_mean"] = pair_mean
    feats["pairwise_dist_cv"]   = pair_cv
    feats["pairwise_dist_p90"]  = pair_p90

    k_nn = min(5, max(1, n - 1))
    knn_mean, knn_cv, knn_p90 = _knn_distance_stats(Xs, k=k_nn)
    feats["knn5_dist_mean"] = knn_mean
    feats["knn5_dist_cv"]   = knn_cv
    feats["knn5_dist_p90"]  = knn_p90

    feats["feature_sparsity"] = float((np.abs(Xs) < 0.01).mean())
    high_corr_frac, corr_disp = _correlation_structure(Xs)
    feats["high_corr_frac"]   = high_corr_frac
    feats["corr_dispersion"]  = corr_disp
    feats["feature_std_dispersion"] = float(np.std(X.std(axis=0)))
    feats["zero_variance_frac"]     = float((X.std(axis=0) < 1e-8).mean())

    raw_std      = X.std(0) + 1e-8
    raw_mean_abs = np.abs(X.mean(0)) + 1e-8
    cv_per_feat  = np.clip(raw_std / raw_mean_abs, 0, 100)
    feats["cv_mean"] = float(np.mean(cv_per_feat))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic cluster-to-class mapping
# ─────────────────────────────────────────────────────────────────────────────
def map_clusters_via_claude(df_features, cluster_labels, class_names, api_key, n_examples=6):
    """
    Call Claude to map each cluster ID to one of the user-supplied class names.

    Sends up to n_examples representative rows from each cluster as context.
    Returns a dict {cluster_id (int): class_name (str)}.
    """
    import anthropic

    unique_clusters = sorted(set(cluster_labels))
    n_cls = len(class_names)

    # Build per-cluster examples
    examples_parts = []
    for cid in unique_clusters:
        mask = cluster_labels == cid
        sample = df_features[mask].head(n_examples)
        examples_parts.append(
            f"Cluster {cid} ({int(mask.sum())} rows):\n{sample.to_string(index=False)}"
        )
    examples_text = "\n\n".join(examples_parts)

    prompt = (
        f"You are a data scientist analyzing clusters from an unsupervised algorithm.\n\n"
        f"The dataset has been split into {len(unique_clusters)} cluster(s).\n"
        f"The possible class names are: {class_names}\n\n"
        f"Here are representative rows from each cluster:\n\n"
        f"{examples_text}\n\n"
        f"Based on the feature values shown, assign each cluster to the most appropriate class name.\n"
        f"Try to assign a different class to each cluster where possible. "
        f"If there are more clusters than classes, multiple clusters may share a class.\n\n"
        f"Respond ONLY with a valid JSON object mapping each cluster ID (as a string) "
        f"to a class name from the list above. Example format:\n"
        f'{{"{unique_clusters[0]}": "{class_names[0]}", ...}}\n'
        f"Output nothing else."
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    mapping_str = json.loads(raw)
    return {int(k): str(v) for k, v in mapping_str.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
def _lse_bar(pred_lse: dict, recommended: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    methods = list(pred_lse.keys())
    vals    = [pred_lse[m] for m in methods]
    colors  = [METHOD_COLORS.get(m, "#888") for m in methods]
    bars = ax.barh(
        [METHOD_DISPLAY.get(m, m) for m in methods],
        vals,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    rec_idx = methods.index(recommended) if recommended in methods else -1
    if rec_idx >= 0:
        bars[rec_idx].set_edgecolor("gold")
        bars[rec_idx].set_linewidth(2.5)

    ax.axvline(CONFIDENCE_FLOOR, color="tomato", linestyle="--", lw=1.2,
               label=f"Confidence floor ({CONFIDENCE_FLOOR})")
    ax.set_xlabel("Predicted LSE  (vs supervised RF baseline)")
    ax.set_title("Expected Label Substitution Efficiency")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(1.05, max(vals) + 0.05))
    plt.tight_layout()
    return fig


def _cluster_size_bar(cluster_labels, class_names_map=None):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    labels = [
        class_names_map.get(int(u), f"Cluster {u}") if class_names_map else f"Cluster {u}"
        for u in unique
    ]
    fig, ax = plt.subplots(figsize=(max(4, len(unique) * 0.9), 3))
    ax.bar(labels, counts,
           color=[METHOD_COLORS.get("kmeans", "#4C72B0")] * len(unique),
           edgecolor="white")
    ax.set_ylabel("Row count")
    ax.set_title("Cluster sizes" + (" → assigned classes" if class_names_map else ""))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Anthropic API Key")
    api_key_input = st.text_input(
        "Paste your API key",
        type="password",
        help="Used only to call Claude for cluster-to-class mapping. Never stored.",
    )
    # Prefer env variable; fallback to user input
    api_key = os.environ.get("ANTHROPIC_API_KEY", "") or api_key_input

    st.markdown("---")
    st.header("About")
    st.markdown(
        """
        **Workflow**
        1. Upload an **unlabeled** CSV
        2. Tell the app how many classes exist and their names
        3. The meta-learner recommends the best clustering method
        4. Run clustering — Claude maps clusters → class names
        5. Download your labeled dataset

        **Label Substitution Efficiency (LSE)**
        ```
        LSE = balanced_acc(pseudo-labels)
              ─────────────────────────────
              balanced_acc(true labels)
        ```
        LSE = 1.0 → as good as human annotation.
        LSE = 0.7 → 70% of supervised accuracy.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔬 Pseudo-Label Advisor")
st.caption(
    "Upload an unlabeled dataset → specify your classes → "
    "get a clustering recommendation → let Claude name your clusters."
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 1 — Upload Unlabeled Dataset")
uploaded = st.file_uploader(
    "Upload a CSV file (features only — no label column needed)",
    type=["csv"],
)

if uploaded is None:
    st.info("Upload a CSV with numeric feature columns to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded)
df_numeric = df_raw.select_dtypes(include=[np.number])

if df_numeric.shape[1] == 0:
    st.error("No numeric feature columns found. Please upload a CSV with numeric data.")
    st.stop()

# Check missing value fraction and impute if necessary
if df_numeric.isnull().any().any():
    total_cells  = df_numeric.shape[0] * df_numeric.shape[1]
    n_missing    = int(df_numeric.isnull().sum().sum())
    missing_frac = n_missing / total_cells

    if missing_frac > 0.05:
        st.warning(
            f"**High missing value rate: {missing_frac:.1%}** ({n_missing} cells). "
            "This exceeds the 5% threshold used during meta-learner training. "
            "Predictions may be less reliable — consider cleaning your data first. "
            "Proceeding with median imputation."
        )
    else:
        st.info(f"Imputed {n_missing} missing values ({missing_frac:.1%}) with column medians.")

    imp = SimpleImputer(strategy="median")
    df_numeric = pd.DataFrame(
        imp.fit_transform(df_numeric), columns=df_numeric.columns
    )

st.success(f"Loaded: **{df_raw.shape[0]} rows** × **{df_numeric.shape[1]} numeric features**")

non_numeric_cols = df_raw.columns.difference(df_numeric.columns).tolist()
if non_numeric_cols:
    st.caption(
        f"Non-numeric columns ignored during clustering: {', '.join(non_numeric_cols)}"
    )

with st.expander("Preview (first 10 rows)"):
    st.dataframe(df_raw.head(10))

X = df_numeric.values


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Class specification
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 2 — Define Your Classes")
st.markdown(
    "Tell the app what you're looking for. "
    "Claude will use these names to label each cluster."
)

col_n, col_names = st.columns([1, 2])
with col_n:
    n_classes = st.number_input(
        "Number of classes", min_value=2, max_value=20, value=3, step=1
    )
with col_names:
    class_names_input = st.text_area(
        "Class names (one per line)",
        value="\n".join([f"Class {i+1}" for i in range(int(n_classes))]),
        height=120,
    )

class_names = [s.strip() for s in class_names_input.strip().split("\n") if s.strip()]

if len(class_names) != int(n_classes):
    st.warning(
        f"You specified {int(n_classes)} classes but entered {len(class_names)} names. "
        "Please make them match."
    )
    st.stop()

st.caption(f"Classes: {' · '.join(f'**{c}**' for c in class_names)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Method recommendation
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 3 — Method Recommendation")

clf_bundle, reg_bundle = load_models()

if clf_bundle is None or reg_bundle is None:
    st.error(
        "No trained models found in `outputs/models/`. "
        "Run notebooks 01 → 05 first to generate `meta_clf_optA.pkl` and `meta_reg_optA.pkl`."
    )
    st.stop()

with st.spinner("Extracting dataset properties..."):
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    feats = extract_features_unlabeled(X, int(n_classes))

clf_feat_cols = clf_bundle["feature_cols"]
reg_feat_cols = reg_bundle["feature_cols"]
lse_cols      = reg_bundle["lse_cols"]
meta_clf      = clf_bundle["pipeline"]
meta_reg      = reg_bundle["pipeline"]

feat_vec_clf = np.array([[feats.get(c, np.nan) for c in clf_feat_cols]])
feat_vec_reg = np.array([[feats.get(c, np.nan) for c in reg_feat_cols]])

recommended     = meta_clf.predict(feat_vec_clf)[0]
pred_lse_vec    = meta_reg.predict(feat_vec_reg)[0]
pred_lse        = dict(zip([c.replace("LSE_", "") for c in lse_cols], pred_lse_vec))

max_pred_lse    = max(pred_lse.values())
low_confidence  = max_pred_lse < CONFIDENCE_FLOOR

rec_col, chart_col = st.columns([1, 2])
with rec_col:
    st.subheader("Recommended Method")
    st.markdown(f"### {METHOD_DISPLAY.get(recommended, recommended)}")
    st.metric("Expected LSE", f"{pred_lse.get(recommended, 0):.3f}")

    if low_confidence:
        st.warning(
            f"⚠️ Max predicted LSE = {max_pred_lse:.2f} < {CONFIDENCE_FLOOR}. "
            "All methods may perform poorly on this dataset. "
            "K-Means is used as a safe fallback."
        )
    else:
        st.success("Within reliable confidence range.")

with chart_col:
    fig_bar = _lse_bar(pred_lse, recommended)
    st.pyplot(fig_bar, use_container_width=False)
    plt.close()

with st.expander("📊 Dataset properties used for recommendation"):
    mf_df = (
        pd.DataFrame.from_dict(feats, orient="index", columns=["value"])
        .rename_axis("meta-feature")
        .round(4)
    )
    mf_df["source"] = mf_df.index.map(
        lambda f: "label-free" if not pd.isna(feats.get(f)) else "imputed (no labels)"
    )
    st.dataframe(mf_df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Run clustering
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 4 — Run Clustering & Assign Class Names")

method_to_run = st.selectbox(
    "Method to run",
    options=list(METHOD_FNS.keys()),
    index=list(METHOD_FNS.keys()).index(recommended),
    format_func=lambda m: (
        f"{METHOD_DISPLAY.get(m, m)}  ← recommended" if m == recommended
        else METHOD_DISPLAY.get(m, m)
    ),
)

if not api_key:
    st.warning(
        "Enter your Anthropic API key in the sidebar to enable cluster → class name mapping."
    )

run_btn = st.button(
    "▶ Run clustering and assign class names",
    type="primary",
    disabled=(not api_key),
)

if run_btn:
    if not api_key:
        st.error("API key required.")
        st.stop()

    # ── Run chosen clustering method ──────────────────────────────────────────
    with st.spinner(f"Running {METHOD_DISPLAY.get(method_to_run, method_to_run)}..."):
        try:
            cluster_fn  = METHOD_FNS[method_to_run]
            cluster_ids = cluster_fn(X_sc, int(n_classes))
        except Exception as e:
            st.error(f"Clustering failed: {e}")
            st.stop()

    unique_clusters = sorted(set(cluster_ids))
    st.success(
        f"Clustering complete — found **{len(unique_clusters)} cluster(s)** "
        f"across {len(cluster_ids)} rows."
    )

    # Cluster size chart (before naming)
    fig_sizes = _cluster_size_bar(cluster_ids)
    st.pyplot(fig_sizes, use_container_width=False)
    plt.close()

    # Per-cluster sample preview
    with st.expander("Cluster samples (before naming)"):
        for cid in unique_clusters:
            mask = cluster_ids == cid
            st.markdown(f"**Cluster {cid}** — {int(mask.sum())} rows")
            st.dataframe(df_numeric[mask].head(6))

    # ── Claude: map cluster IDs → class names ─────────────────────────────────
    with st.spinner("Calling Claude to assign class names to clusters..."):
        try:
            cluster_to_class = map_clusters_via_claude(
                df_features=df_numeric,
                cluster_labels=cluster_ids,
                class_names=class_names,
                api_key=api_key,
                n_examples=6,
            )
        except Exception as e:
            st.error(f"Claude API call failed: {e}")
            st.stop()

    st.success("Class names assigned!")

    # Show the mapping Claude produced
    mapping_df = pd.DataFrame([
        {
            "Cluster ID": cid,
            "Assigned class": cluster_to_class.get(cid, "unknown"),
            "Row count": int((cluster_ids == cid).sum()),
        }
        for cid in unique_clusters
    ])
    st.table(mapping_df)

    # Cluster size chart with names
    fig_named = _cluster_size_bar(cluster_ids, cluster_to_class)
    st.pyplot(fig_named, use_container_width=False)
    plt.close()

    # ── Build labeled output dataset ──────────────────────────────────────────
    df_out = df_raw.copy()
    df_out["cluster_id"]    = cluster_ids
    df_out["predicted_class"] = [
        cluster_to_class.get(int(c), f"cluster_{c}") for c in cluster_ids
    ]

    st.subheader("Labeled Dataset Preview")
    st.dataframe(df_out.head(20))

    # Per-class sample previews
    with st.expander("Sample rows per predicted class"):
        for cls in class_names:
            subset = df_out[df_out["predicted_class"] == cls]
            if len(subset) == 0:
                continue
            pct = 100 * len(subset) / len(df_out)
            st.markdown(f"**{cls}** — {len(subset)} rows ({pct:.1f}%)")
            st.dataframe(subset.head(5))

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download labeled CSV",
        data=csv_bytes,
        file_name=f"labeled_{uploaded.name}",
        mime="text/csv",
    )
