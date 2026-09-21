#!/usr/bin/env python3
# get_lof_blast_prevalence.py
# Re-loads the same BLAST110 dataset lof.py builds (data loading only, no training)
# to recover the blast prevalence used in the nested CV run.

import os
import re
import pandas as pd
from FlowCytometryTools import FCMeasurement

# ==================== CONFIG (must match lof.py exactly) ====================
FCS_FOLDER   = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
LABEL_FOLDER = "/storage/mezya.sezen/mphasis/dataset/labels/"
FEATURES = ["SSC-A", "Horizon_V450-A", "Horizon_V500-A", "PerCP-A", "PC7-A"]

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

def sample_events(df, n=5000):
    return df.sample(n=n, random_state=42) if len(df) > n else df

def extract_key(filename):
    m = re.match(r"(BLAST\d+_\d+_P\d+)", filename)
    return m.group(1) if m else None

print("[INFO] Rebuilding BLAST110 dataset exactly as lof.py does (no model training)...")

fcs_files = [f for f in os.listdir(FCS_FOLDER) if f.endswith(".fcs")]
label_files = [f for f in os.listdir(LABEL_FOLDER) if f.endswith(".csv")]
label_map = {extract_key(f): f for f in label_files}

all_rows = []
for fcs_name in fcs_files:
    key = extract_key(fcs_name)
    if key is None or key not in label_map:
        continue

    fcs_path = os.path.join(FCS_FOLDER, fcs_name)
    csv_path = os.path.join(LABEL_FOLDER, label_map[key])

    df_fcs = load_fcs_stable(fcs_path)
    labels_df = pd.read_csv(csv_path)
    merged = df_fcs.merge(labels_df[["event_ID", "Blast"]], on="event_ID")
    merged = sample_events(merged, 5000)
    merged["sample_id"] = key
    merged["patient_id"] = "_".join(key.split("_")[:2])
    all_rows.append(merged)

if not all_rows:
    raise RuntimeError("No valid BLAST110 samples found! Check FCS_FOLDER/LABEL_FOLDER paths.")

df_blast110 = pd.concat(all_rows, ignore_index=True)
df_blast110 = df_blast110[df_blast110["Blast"].isin([0, 1])].reset_index(drop=True)

print(f"\n[RESULT]")
print(f"  Total cells:        {len(df_blast110):,}")
print(f"  Blast prevalence:   {df_blast110['Blast'].mean():.3%}")
print(f"  Patients:           {df_blast110['patient_id'].nunique()}")
print(f"  Samples (files):    {df_blast110['sample_id'].nunique()}")

# Save a small summary CSV for the record / manuscript
summary_path = "/storage/mezya.sezen/mphasis/lof_results/lof_21_2_26/lof_dataset_blast_prevalence.csv"
pd.DataFrame([{
    "total_cells": len(df_blast110),
    "blast_prevalence": df_blast110["Blast"].mean(),
    "n_patients": df_blast110["patient_id"].nunique(),
    "n_samples": df_blast110["sample_id"].nunique()
}]).to_csv(summary_path, index=False)
print(f"\n[SAVED] {summary_path}")