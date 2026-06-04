import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier

# 10kb total per side
BINS_PER_SIDE = 200
BIN_BP = 50


def _n_bins(kb: float) -> int:
    return round(kb * 1000 / BIN_BP)


def extract_features(k9_array: np.ndarray, prefix: str) -> dict:
    """
    Compute mean K9 signal over cumulative windows off each TE edge.
    Returns a dict with mean K9 signal from 1-10kb on each side.
    """
    left  = k9_array[:BINS_PER_SIDE]   # bins[0:200]   — left flank
    right = k9_array[BINS_PER_SIDE:]   # bins[200:400]  — right flank

    feats = {}
    for kb in [1,2,3,4,5,6,7,8,9,10]:
        n = _n_bins(kb)
        feats[f"{prefix}_mean_{kb}kb_left"]  = float(left[-n:].mean())
        feats[f"{prefix}_mean_{kb}kb_right"] = float(right[:n].mean())

    return feats


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature DataFrame from a full_df.
    Returns a DataFrame with shape (len(df), n_features).
    """
    rows = []
    for _, row in df.iterrows():
        feats = {}
        feats.update(extract_features(row["A4_K9"], "A4"))
        feats.update(extract_features(row["A7_K9"], "A7"))
        rows.append(feats)
    return pd.DataFrame(rows)

def train_random_forest(train_df, val_df, test_df):
    """
    Train a multi-output Random Forest on K9 features (A4 and A7 outputs)
    Returns (model, test_metrics).
    """
    X_train = build_feature_matrix(train_df)
    X_val   = build_feature_matrix(val_df)
    X_test  = build_feature_matrix(test_df)
    y_train = train_df[["label_A4", "label_A7"]].values
    y_val   = val_df  [["label_A4", "label_A7"]].values
    y_test  = test_df [["label_A4", "label_A7"]].values

    print(f"Feature matrix: {X_train.shape[0]} train  |  {X_val.shape[0]} val  |  {X_test.shape[0]} test  |  {X_train.shape[1]} features")

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42), n_jobs=-1)
    model.fit(X_train, y_train)
    evaluate_rf(model, X_train, y_train, "Train")
    evaluate_rf(model, X_val,   y_val,   "Val")
    test_metrics = evaluate_rf(model, X_test, y_test, "Test")

    return model, test_metrics


def evaluate_rf(model, X, y, split_name: str) -> dict:
    """
    Evaluate the trained model and print per-head auROC and bAcc.
    Returns a dict with A4 and A7 auROC and bAcc.
    """
    proba = np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])
    preds = (proba >= 0.5).astype(int)

    metrics = {}
    lines = []
    for i, strain in enumerate(["A4", "A7"]):
        auroc = (
            roc_auc_score(y[:, i], proba[:, i])
            if len(np.unique(y[:, i])) > 1 else float("nan")
        )
        bacc = balanced_accuracy_score(y[:, i], preds[:, i])
        metrics[f"auROC_{strain}"] = auroc
        metrics[f"bAcc_{strain}"]  = bacc
        lines.append(f"  {strain}: auROC={auroc:.3f}  bAcc={bacc:.3f}")

    mean_auroc = np.nanmean([metrics["auROC_A4"], metrics["auROC_A7"]])
    print(f"\n{split_name}")
    print("\n".join(lines))
    print(f"  mean auROC: {mean_auroc:.3f}")
    return metrics


def feature_importance(model, feature_names, top_n: int = 15):
    """
    Print the top_n most important features, averaged across both RF heads (A4 and A7).
    """
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    top_idx = np.argsort(importances)[::-1][:top_n]
    print(f"\nTop {top_n} features (averaged across A4/A7 heads):")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:<40s} {importances[i]:.4f}")


if __name__ == "__main__":
    from load_df import load_K9, load_TE_map, build_te_df, sample_negatives, build_full_df
    from load_tensors import make_splits

    DATA_DIR = Path(__file__).parent.parent / "test_data"

    A4_k9    = load_K9(str(DATA_DIR / "A4_K9.bed"))
    A7_k9    = load_K9(str(DATA_DIR / "A7_K9.bed"))
    A4_to_A7 = load_TE_map(str(DATA_DIR / "A4_to_A7_map.txt"))
    A7_to_A4 = load_TE_map(str(DATA_DIR / "A7_to_A4_map.txt"))

    te_df   = build_te_df(A4_k9, A7_k9, A4_to_A7, A7_to_A4)
    neg_df  = sample_negatives(te_df, A4_k9, A7_k9, neg_ratio=0.10)
    full_df = build_full_df(te_df, neg_df)

    train_df, val_df, test_df = make_splits(full_df)
    model, test_metrics = train_random_forest(train_df, val_df, test_df)
    feature_importance(model, build_feature_matrix(train_df).columns)

