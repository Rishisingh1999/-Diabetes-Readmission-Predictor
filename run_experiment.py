#!/usr/bin/env python3
"""
Diabetes Readmission Predictor - Reproducible Experiment
Target: readmitted == '<30' (binary 30-day readmission)
Dataset: UCI Diabetes 130-US hospitals (1999-2008)
"""
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline

SEED = 42
np.random.seed(SEED)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
DATA_PATH = Path("diabetic_data.csv")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    file_hash = sha256_file(DATA_PATH)
    print(f"Shape: {df.shape}, SHA256: {file_hash}")

    # Target: 30-day readmission
    df["target"] = (df["readmitted"] == "<30").astype(int)
    print("Target prevalence (<30):", df["target"].mean().round(4))
    print(df["readmitted"].value_counts())

    # Age ordinal encoding
    age_order = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    age_map = {a: i for i, a in enumerate(age_order)}
    df["age_ord"] = df["age"].map(age_map).fillna(-1).astype(int)

    features = [
        "age_ord", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "number_diagnoses"
    ]
    # rename for clarity in outputs
    feature_names = [
        "age", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "number_diagnoses"
    ]

    X = df[features].copy()
    y = df["target"].values
    groups = df["patient_nbr"].values

    # Patient-grouped split to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_trainval, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_trainval, y_test = y[train_idx], y[test_idx]
    groups_trainval = groups[train_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx2, val_idx = next(gss2.split(X_trainval, y_trainval, groups_trainval))
    X_train, X_val = X_trainval.iloc[train_idx2], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval[train_idx2], y_trainval[val_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=SEED,
            max_iter=1000,
            solver="lbfgs"
        ))
    ])
    pipe.fit(X_train, y_train)

    # Coefficients (on scaled features)
    coefs = dict(zip(feature_names, pipe.named_steps["clf"].coef_[0].round(4).tolist()))
    intercept = float(pipe.named_steps["clf"].intercept_[0])

    # Validation threshold tuning for F1
    val_proba = pipe.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1s)
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    print(f"Best val F1 threshold: {best_thr:.4f}")

    def eval_at_threshold(y_true, proba, threshold):
        pred = (proba >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "average_precision": float(average_precision_score(y_true, proba)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, pred).tolist()
        }

    test_proba = pipe.predict_proba(X_test)[:, 1]
    metrics_default = eval_at_threshold(y_test, test_proba, 0.5)
    metrics_tuned = eval_at_threshold(y_test, test_proba, best_thr)

    print("\n=== Test metrics @ 0.5 ===")
    print(json.dumps(metrics_default, indent=2))
    print("\n=== Test metrics @ tuned thr ===")
    print(json.dumps(metrics_tuned, indent=2))

    # Save predictions
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_proba": test_proba,
        "y_pred_0.5": (test_proba >= 0.5).astype(int),
        "y_pred_tuned": (test_proba >= best_thr).astype(int)
    })
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # Plots
    # Correlation heatmap
    corr_df = X_train.copy()
    corr_df.columns = feature_names
    corr_df["target"] = y_train
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap (Train)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()

    # Coefficients
    coef_series = pd.Series(coefs).sort_values()
    plt.figure(figsize=(8, 5))
    coef_series.plot(kind="barh", color="steelblue")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Logistic Regression Coefficients (standardized features)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_coefficients.png", dpi=120)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics_default['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=120)
    plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP = {metrics_default['average_precision']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curve.png", dpi=120)
    plt.close()

    # Confusion matrix (default thr)
    cm = np.array(metrics_default["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix (Test, thr=0.5)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # Metrics JSON
    report = {
        "target_definition": "readmitted == '<30' (1); NO and >30 = 0",
        "dataset_source": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "kaggle_mirror": "https://www.kaggle.com/datasets/brandao/diabetes",
        "dataset_rows": int(df.shape[0]),
        "dataset_cols": int(df.shape[1]),
        "features": feature_names,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "split_method": "GroupShuffleSplit by patient_nbr (prevents patient leakage)",
        "model": "LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)",
        "threshold_default": 0.5,
        "threshold_tuned_on_val_for_f1": float(round(best_thr, 4)),
        "test_metrics_default_thr": {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in metrics_default.items()},
        "test_metrics_tuned_thr": {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in metrics_tuned.items()},
        "coefficients": coefs,
        "intercept": round(intercept, 4),
        "file_sha256": file_hash,
        "run_date": datetime.now(timezone.utc).isoformat(),
        "seed": SEED
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved all artifacts to", OUT_DIR)

if __name__ == "__main__":
    main()
