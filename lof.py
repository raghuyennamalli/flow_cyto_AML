# -*- coding: utf-8 -*-
import os
import glob
import time
import pandas as pd
import numpy as np
from FlowCytometryTools import FCMeasurement
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# ===============================================================
# Helper
# ===============================================================
def timestamp(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ===============================================================
# Correct FCS Loader (uses $PnS names)
# ===============================================================
def load_fcs_stable(file_path):
    """
    Load FCS using FlowCytometryTools and rename columns
    based on proper marker names ($PnS).
    """
    sample = FCMeasurement(ID=file_path, datafile=file_path)
    df = sample.data.copy()
    meta = sample.meta
    new_cols = []

    for i, col in enumerate(df.columns, start=1):
        marker = meta.get(f"$P{i}S", "").strip()
        detector = col.strip()

        # If marker exists, use it
        if marker not in ["", " ", None]:
            clean_name = marker.replace(" ", "_")
        else:
            clean_name = detector.replace(" ", "_")

        new_cols.append(clean_name)

    df.columns = new_cols
    return df

def sample_events(df, n=5000):
    if len(df) > n:
        return df.sample(n=n, random_state=42)
    return df

# ===============================================================
# Folder paths
# ===============================================================
healthy_folder = "/storage/mezya.sezen/mphasis/dataset/after_scaling/"
all_samples_folder = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
labels_folder = "/storage/mezya.sezen/mphasis/dataset/labels/"

# ===============================================================
# 1. Train LOF on healthy samples
# ===============================================================
timestamp("Loading and training on healthy samples...")

healthy_files = glob.glob(os.path.join(healthy_folder, "*.fcs"))
dfs_healthy = []

for f in healthy_files:
    try:
        df = load_fcs_stable(f)
        df = sample_events(df, 5000)
        dfs_healthy.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

df_healthy = pd.concat(dfs_healthy, ignore_index=True)

# use ALL markers except time
features = [col for col in df_healthy.columns if "Time" not in col]
print(f"Using features: {features}")

X_healthy = df_healthy[features].values

lof = LocalOutlierFactor(n_neighbors=50, contamination=0.01, novelty=True)
start_time = time.time()
lof.fit(X_healthy)
end_time = time.time()
timestamp(f"LOF trained in {end_time - start_time:.2f} seconds.")
joblib.dump(lof, "lof_blast_model.pkl")

# ===============================================================
# 2. Test model on all samples
# ===============================================================
timestamp("Predicting and evaluating on all samples...")

precisions, recalls, f1s = [], [], []
all_files = glob.glob(os.path.join(all_samples_folder, "*.fcs"))

for fcs_file in all_files:
    file_base = os.path.basename(fcs_file)
    name = file_base.replace(".fcs", "")

    pattern = os.path.join(labels_folder, f"{name}_*.csv")
    matching = glob.glob(pattern)
    if len(matching) == 0:
        print(f"No label file found for {name}, skipping.")
        continue

    label_file = matching[0]
    print(f"Using label file: {label_file}")

    try:
        df = load_fcs_stable(fcs_file)
        df = sample_events(df, 5000)

        X_all = df[[f for f in features if f in df.columns]].values

        df_labels = pd.read_csv(label_file)
        if "Blast" not in df_labels.columns:
            print(f"No 'Blast' column in {label_file}, skipping.")
            continue

        y_true = df_labels["Blast"].values[:len(df)]
        preds = lof.predict(X_all)
        preds = np.where(preds == -1, 1, 0)

        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))

    except Exception as e:
        print(f"Error in {name}: {e}")

# ===============================================================
# 3. Save Metrics
# ===============================================================
if precisions and recalls and f1s:
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    print("\nAverage Metrics Across All Samples:")
    print(f"Precision: {avg_precision:.3f}")
    print(f"Recall:    {avg_recall:.3f}")
    print(f"F1 Score:  {avg_f1:.3f}")

    df_avg = pd.DataFrame([{
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1
    }])
    df_avg.to_csv("lof_average_metrics.csv", index=False)
    timestamp("Saved average evaluation metrics.")
else:
    timestamp("No metrics computed. Check label folder paths or Blast column.")
