"""Dataset loading and preprocessing helpers for the benchmark pipeline."""

import numpy as np
import openml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

SEED = 42
MIN_GT_LIFT_OVER_MAJORITY = 0.05


def _one_hot_encoder():
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_tabular_preprocessor(X):
    """
    Build preprocessing that keeps numeric and categorical predictors.

    Numeric columns get median imputation. Categorical columns get most-frequent
    imputation plus one-hot encoding. The returned transformer outputs dense
    numeric arrays so downstream clustering code can use standard sklearn APIs.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("dataset has no usable feature columns")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_openml_classification(dataset_id, test_size=0.2, random_state=SEED):
    """
    Load one OpenML classification dataset with categorical-safe preprocessing.

    Returns transformed train/test arrays, encoded labels, and a small metadata
    dictionary. Labels are encoded only for evaluation; the feature transformer
    never receives labels.
    """
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    if y is None:
        raise ValueError(f"dataset {dataset_id} has no default target")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    preprocessor = build_tabular_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    metadata = {
        "dataset_id": int(dataset_id),
        "name": getattr(ds, "name", str(dataset_id)),
        "n_instances": int(len(X)),
        "n_raw_features": int(X.shape[1]),
        "n_transformed_features": int(X_train.shape[1]),
        "n_classes": int(len(label_encoder.classes_)),
        "target": ds.default_target_attribute,
        "label_encoder": label_encoder,
        "preprocessor": preprocessor,
    }
    return X_train, X_test, y_train, y_test, metadata


def groundtruth_quality_gate(X_train, y_train, X_test, y_test, min_lift=MIN_GT_LIFT_OVER_MAJORITY):
    """
    Decide whether a dataset is learnable enough for meaningful LSE.

    Datasets where the ground-truth RF barely beats majority class can make LSE
    look strong even when pseudo-labels only reproduce majority behavior.
    """
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X_train, y_train)
    gt_acc = float(rf.score(X_test, y_test))

    _, counts = np.unique(y_test, return_counts=True)
    majority_acc = float(counts.max() / counts.sum())
    lift = gt_acc - majority_acc

    return {
        "keep": bool(lift >= min_lift),
        "gt_accuracy": gt_acc,
        "majority_accuracy": majority_acc,
        "gt_lift_over_majority": lift,
        "min_required_lift": min_lift,
        "failure_reason": "" if lift >= min_lift else "gt_near_majority",
    }
