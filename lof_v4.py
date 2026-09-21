#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from FlowCytometryTools import FCMeasurement
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter


# ==================== CONFIG ====================
FCS_FOLDER   = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
LABEL_FOLDER = "/storage/mezya.sezen/mphasis/dataset/labels/"
OUTPUT_FOLDER = "/storage/mezya.sezen/mphasis/lof_results/lof_supervised_5m"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Use the same 4 blast-related features you’ve been using
FEATURES = ["SSC-A","Horizon_V450-A","Horizon_V500-A", "PerCP-A", "PC7-A"]

# Hyperparameter grid for LOF
PARAM_GRID = {
    "n_neighbors": [10, 20, 50, 100],
    "contamination": [0.01, 0.025, 0.05, 0.1]
}

OUTER_SPLITS = 5
INNER_SPLITS = 3

# ==================== HELPERS ====================
def load_fcs_stable(path):
    """Load FCS and return pandas DataFrame with clean column names."""
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

def sample_events(df, n=5000):
    return df.sample(n=n, random_state=42) if len(df) > n else df

def extract_key(filename):
    """
    From BLAST110_41_P1_Cleaned_transformed_scaled.fcs
    get BLAST110_41_P1 (to match BLAST110_41_P1.csv)
    """
    m = re.match(r"(BLAST\d+_\d+_P\d+)", filename)
    return m.group(1) if m else None

# ==================== LOF WRAPPER (supervised use) ====================
class LOFWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrap LocalOutlierFactor so it behaves like a classifier:
      - fit(X, y) ignores y (unsupervised).
      - predict(X) returns 1 for outliers (blast), 0 for inliers.
    """
    def __init__(self, n_neighbors=20, contamination=0.01):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model_ = None

    def fit(self, X, y=None):
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True
        )
        self.model_.fit(X)
        return self

    def predict(self, X):
        # LOF: -1 = outlier, 1 = inlier
        preds = self.model_.predict(X)
        return np.where(preds == -1, 1, 0)  # 1=blast, 0=non-blast

# ==================== BUILD BLAST110 DATASET ====================
print("Building labeled BLAST110 dataset...")

all_rows = []

fcs_files = [f for f in os.listdir(FCS_FOLDER) if f.endswith(".fcs")]
label_files = [f for f in os.listdir(LABEL_FOLDER) if f.endswith(".csv")]
label_map = {extract_key(f): f for f in label_files}

for fcs_name in fcs_files:
    key = extract_key(fcs_name)
    if key is None:
        print(f"[WARN] Could not extract key from {fcs_name}")
        continue
    if key not in label_map:
        print(f"[WARN] No matching label CSV for {fcs_name} (key={key})")
        continue

    fcs_path = os.path.join(FCS_FOLDER, fcs_name)
    csv_path = os.path.join(LABEL_FOLDER, label_map[key])

    # load FCS and labels
    df_fcs = load_fcs_stable(fcs_path)
    df_fcs = sample_events(df_fcs, 5000)   # 5K per sample for speed
    n_cells = len(df_fcs)

    labels_df = pd.read_csv(csv_path)
    if "Blast" not in labels_df.columns:
        print(f"[WARN] No 'Blast' column in {csv_path}")
        continue
    # align sizes
    y = labels_df["Blast"].values[:n_cells]

    # attach label and patient/sample ids
    df_fcs["Blast"] = y
    df_fcs["sample_id"] = key
    df_fcs["patient_id"] = "_".join(key.split("_")[:2])

    all_rows.append(df_fcs)

if not all_rows:
    raise RuntimeError("No BLAST110 samples were successfully merged; check paths/names.")

df_blast110 = pd.concat(all_rows, ignore_index=True)
print("Total cells:", len(df_blast110))
print("Columns available:", df_blast110.columns.tolist())

# keep only samples where Blast is 0 or 1
df_blast110 = df_blast110[df_blast110["Blast"].isin([0,1])]
df_blast110 = df_blast110.reset_index(drop=True)

# ==================== PREPARE ARRAYS ====================
missing = [f for f in FEATURES if f not in df_blast110.columns]
if missing:
    raise KeyError(f"Missing required features in data: {missing}")

X = df_blast110[FEATURES].values
y = df_blast110["Blast"].values.astype(int)
groups = df_blast110["patient_id"].values
sample_ids = df_blast110["sample_id"].values

# ==================== NESTED GROUPKFold + GRIDSEARCH ====================
outer_cv = GroupKFold(n_splits=OUTER_SPLITS)
inner_cv = GroupKFold(n_splits=INNER_SPLITS)

fold_metrics = []
best_params_each_fold = []

print("\nStarting nested GroupKFold supervised LOF tuning...")

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
    print(f"\n=== OUTER FOLD {fold}/{OUTER_SPLITS} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Inner CV: tune LOF hyperparameters on training partition
    gs = GridSearchCV(
        LOFWrapper(),
        PARAM_GRID,
        cv=inner_cv.split(X_train, y_train, groups_train),
        scoring="f1",
        refit=True,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)   # y used for scoring (F1)
    best_model = gs.best_estimator_
    best_params_each_fold.append(gs.best_params_)

    print("Best params (inner):", gs.best_params_)

    # Evaluate on outer test partition
    y_pred = best_model.predict(X_test)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"Fold {fold} -> Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    fold_metrics.append({
        "fold": fold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "best_n_neighbors": gs.best_params_["n_neighbors"],
        "best_contamination": gs.best_params_["contamination"]
    })

# Save per-fold metrics
df_fold_metrics = pd.DataFrame(fold_metrics)
df_fold_metrics.to_csv(os.path.join(OUTPUT_FOLDER, "lof_supervised_nestedCV_folds.csv"), index=False)

print("\nMean metrics across outer folds:")
print(df_fold_metrics[["precision", "recall", "f1"]].mean())

# ==================== TRAIN FINAL LOF ON FULL BLAST110 ====================
# Choose majority params across folds
nn_majority = Counter([p["n_neighbors"] for p in best_params_each_fold]).most_common(1)[0][0]
cont_majority = Counter([p["contamination"] for p in best_params_each_fold]).most_common(1)[0][0]

print(f"\nFinal chosen params (majority vote across folds): "
      f"n_neighbors={nn_majority}, contamination={cont_majority}")

final_lof = LOFWrapper(n_neighbors=nn_majority, contamination=cont_majority)
final_lof.fit(X, y)

# Evaluate on full BLAST110 (for reference)
y_pred_full = final_lof.predict(X)
prec_full = precision_score(y, y_pred_full, zero_division=0)
rec_full  = recall_score(y, y_pred_full, zero_division=0)
f1_full   = f1_score(y, y_pred_full, zero_division=0)

with open(os.path.join(OUTPUT_FOLDER, "lof_supervised_final_metrics.txt"), "w") as fh:
    fh.write(f"Final LOF (full BLAST110) Precision={prec_full:.4f}\n")
    fh.write(f"Final LOF (full BLAST110) Recall={rec_full:.4f}\n")
    fh.write(f"Final LOF (full BLAST110) F1={f1_full:.4f}\n")

print("\nFinal LOF on full BLAST110:")
print(f"Precision={prec_full:.3f}, Recall={rec_full:.3f}, F1={f1_full:.3f}")

# ==================== PLOT PER-FOLD METRICS ====================
plt.figure(figsize=(8,5))
x = np.arange(len(df_fold_metrics["fold"]))
width = 0.25

plt.bar(x - width, df_fold_metrics["precision"], width=width, label="Precision")
plt.bar(x,         df_fold_metrics["recall"],    width=width, label="Recall")
plt.bar(x + width, df_fold_metrics["f1"],        width=width, label="F1")

plt.xticks(x, [f"Fold {f}" for f in df_fold_metrics["fold"]])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("LOF supervised nested CV metrics per fold (BLAST110)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "lof_supervised_nestedCV_metrics_per_fold.png"), dpi=150)
plt.close()

print(f"\nSaved per-fold metrics plot to {os.path.join(OUTPUT_FOLDER, 'lof_supervised_nestedCV_metrics_per_fold.png')}")
