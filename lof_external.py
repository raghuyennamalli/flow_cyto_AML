#!/usr/bin/env python3
# lof_external_validation.py
# External validation of the saved final LOF model on LAIP29 (diagnosis + follow-up samples)

import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from FlowCytometryTools import FCMeasurement
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    cohen_kappa_score, roc_curve, precision_recall_curve
)

# ==================================================
# PATHS (edit these)
# ==================================================
LOF_OUTPUT_FOLDER = "/storage/mezya.sezen/mphasis/lof_results/lof_21_2_26"
LAIP29_FCS_PATH = "/storage/mezya.sezen/mphasis/LAIP29/scaled/"
LAIP29_LABEL_PATH = "/storage/mezya.sezen/mphasis/LAIP29/labels/"
OUTPUT_PATH = os.path.join(LOF_OUTPUT_FOLDER, "external_validation_LAIP29")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Must match training feature order exactly (note underscores, matching lof.py)
FEATURES = ["SSC-A", "Horizon_V450-A", "Horizon_V500-A", "PerCP-A", "PC7-A"]

EXTERNAL_SAMPLES = {
    "diagnosis": "LAIP29_8_Dx_P2",
    "follow_up": "LAIP29_9_FU_P3",
}

# ==================================================
# Utility: read FCS -> pandas dataframe (same as lof.py's load_fcs_stable)
# ==================================================
def load_fcs_stable(path):
    sample = FCMeasurement(ID=os.path.basename(path), datafile=path)
    df = sample.data.copy()
    try:
        meta_channels = sample.meta.get("_channels_", None)
        if meta_channels is not None and "$PnN" in meta_channels:
            cols = [c.replace(" ", "_") for c in meta_channels["$PnN"].tolist()]
        else:
            cols = [c.replace(" ", "_") for c in df.columns]
    except Exception:
        cols = [c.replace(" ", "_") for c in df.columns]
    df.columns = cols
    return df


def load_external_sample(sample_key):
    fcs_candidates = glob.glob(os.path.join(LAIP29_FCS_PATH, f"{sample_key}*.fcs"))
    if not fcs_candidates:
        raise FileNotFoundError(f"No FCS file found for {sample_key} in {LAIP29_FCS_PATH}")
    fcs_path = fcs_candidates[0]

    label_candidates = glob.glob(os.path.join(LAIP29_LABEL_PATH, f"{sample_key}*.csv"))
    if not label_candidates:
        raise FileNotFoundError(f"No label file found for {sample_key} in {LAIP29_LABEL_PATH}")
    label_path = label_candidates[0]

    df_fcs = load_fcs_stable(fcs_path)
    labels_df = pd.read_csv(label_path)
    merged = df_fcs.merge(labels_df[["event_ID", "Blast"]], on="event_ID")

    missing = [f for f in FEATURES if f not in merged.columns]
    if missing:
        raise KeyError(f"Missing features {missing} in {sample_key}")

    X = merged[FEATURES].values
    y = merged["Blast"].values.astype(int)
    return X, y, merged


def evaluate_and_plot(y_true, y_scores, tau, sample_name, model_tag, lower_is_anomalous):
    """
    lower_is_anomalous: True if using sklearn LOF's native decision_function
                         (lower/negative = more anomalous, i.e. blast).
    """
    if lower_is_anomalous:
        y_pred = (y_scores < tau).astype(int)
        # For AUROC/AUPRC, flip sign so higher = more anomalous (positive class)
        scores_for_auc = -y_scores
    else:
        y_pred = (y_scores >= tau).astype(int)
        scores_for_auc = y_scores

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics = {
        "sample": sample_name,
        "model": model_tag,
        "n_events": len(y_true),
        "blast_prevalence": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "auroc": roc_auc_score(y_true, scores_for_auc) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": average_precision_score(y_true, scores_for_auc) if len(np.unique(y_true)) > 1 else np.nan,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "tau_used": tau,
    }

    if len(np.unique(y_true)) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(y_true, scores_for_auc)
        axes[0].plot(fpr, tpr, lw=2, label=f"AUROC={metrics['auroc']:.3f}")
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"{model_tag} ROC - {sample_name}")
        axes[0].legend()

        prec, rec, _ = precision_recall_curve(y_true, scores_for_auc)
        axes[1].plot(rec, prec, lw=2, label=f"AUPRC={metrics['auprc']:.3f}")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title(f"{model_tag} PR - {sample_name}")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f"{model_tag}_{sample_name}_roc_pr.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xticks([0, 1], ['WBC', 'Blast']); plt.yticks([0, 1], ['WBC', 'Blast'])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{model_tag} Confusion Matrix - {sample_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"{model_tag}_{sample_name}_confusion_matrix.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    return metrics


# ==================================================
# MAIN
# ==================================================
model_path = os.path.join(LOF_OUTPUT_FOLDER, "lof_final_model.joblib")
threshold_path = os.path.join(LOF_OUTPUT_FOLDER, "lof_threshold.npy")
metadata_path = os.path.join(LOF_OUTPUT_FOLDER, "lof_metadata.pkl")

if not all(os.path.exists(p) for p in [model_path, threshold_path, metadata_path]):
    raise FileNotFoundError("Saved LOF model/threshold/metadata not found. Check LOF_OUTPUT_FOLDER.")

final_lof = joblib.load(model_path)
saved_threshold = float(np.load(threshold_path))
metadata = joblib.load(metadata_path)

print(f"[INFO] Loaded LOF model: n_neighbors={metadata['n_neighbors']}, "
      f"contamination={metadata['contamination']}, threshold={metadata['threshold']:.4f}")

# NOTE: in lof.py, decision_function score < threshold => blast (outlier),
# matching optimize_lof_threshold()'s convention (lower = more anomalous).
all_results = []

for label, sample_key in EXTERNAL_SAMPLES.items():
    print(f"\n[INFO] Loading external sample: {sample_key} ({label})")
    X_ext, y_ext, merged_df = load_external_sample(sample_key)
    print(f"        n_events={len(y_ext)}, blast_prevalence={np.mean(y_ext):.3%}")

    scores_ext = final_lof.decision_function(X_ext)

    m = evaluate_and_plot(
        y_ext, scores_ext, saved_threshold,
        sample_key, "LOF_final", lower_is_anomalous=True
    )
    all_results.append(m)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_PATH, "LOF_external_validation_LAIP29.csv"), index=False)

print("\n=== LOF EXTERNAL VALIDATION COMPLETE ===")
print(results_df[["sample", "model", "n_events", "blast_prevalence",
                   "accuracy", "precision", "recall", "f1", "auroc", "auprc", "kappa"]])
print(f"\n[OUTPUT] {OUTPUT_PATH}")