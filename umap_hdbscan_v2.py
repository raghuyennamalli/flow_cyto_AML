#!/usr/bin/env python3
# =============================================================
# UMAP + HDBSCAN clustering workflow (HPC optimized) - 5 FEATURES
# =============================================================

import os
import glob
import time
import datetime
import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fcsparser import parse
import pickle

# =============================================================
# CONFIGURATION
# =============================================================

N_JOBS = 10     # match PBS ppn=10
EVENTS_PER_FILE = 5000
FEATURES = ["SSC-A", "CD45 KO", "CD34 Cy55", "CD117", "CD13 BV421"]

OUTPUT_DIR = "/storage/mezya.sezen/mphasis/umap_hdbscan/5feat_results/"

# =============================================================
# Utility: timestamp
# =============================================================

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================
# Step 1: Load all FCS files
# =============================================================

def load_all_fcs(folder_path):
    start = time.time()
    print(f"[{now()}] Starting to load FCS files...")

    all_files = glob.glob(os.path.join(folder_path, "*.fcs"))
    dfs = []

    for file in all_files:
        print(f"[{now()}] Reading file: {file}")
        try:
            meta, data = parse(file, reformat_meta=True)
            sample_id = os.path.basename(file).replace(".fcs", "")
            data["sample_id"] = sample_id
            dfs.append(data)
        except Exception as e:
            print(f"[{now()}] Error reading {file}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[{now()}] Combined {len(all_files)} files with {len(df)} events.")
    print(f"[{now()}] Time taken to load: {round((time.time() - start)/60, 2)} min")

    return df


# =============================================================
# Step 2: Sample each file to fixed number of events
# =============================================================

def sample_events_per_file(df, sample_col="sample_id", n_events=5000):
    sampled = []
    for sid in df[sample_col].unique():
        block = df[df[sample_col] == sid]
        if len(block) >= n_events:
            sampled.append(block.sample(n_events, random_state=42))
        else:
            sampled.append(block)
    return pd.concat(sampled).reset_index(drop=True)


# =============================================================
# MAIN WORKFLOW
# =============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Load FCS files
    # ---------------------------------------------------------
    folder = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
    df = load_all_fcs(folder)

    df.to_pickle(os.path.join(OUTPUT_DIR, "blast110_full_dataset_5feat.pkl"))

    # ---------------------------------------------------------
    # Downsample each file
    # ---------------------------------------------------------
    print(f"[{now()}] Sampling events...")
    df = sample_events_per_file(df, n_events=EVENTS_PER_FILE)
    print(f"[{now()}] After sampling: {len(df)} events from {df['sample_id'].nunique()} samples")

    # ---------------------------------------------------------
    # Clean feature space
    # ---------------------------------------------------------
    print(f"[{now()}] Cleaning NaNs and infs...")
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=FEATURES)
    after = len(df)
    print(f"[{now()}] Dropped {before-after} rows")

    # ---------------------------------------------------------
    # Prepare arrays
    # ---------------------------------------------------------
    X = df[FEATURES].values
    groups = df["sample_id"].values

    # ---------------------------------------------------------
    # 5-Fold GroupKFold
    # ---------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    overall_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        fold_start = time.time()
        print(f"\n[{now()}] ===== Fold {fold+1} started =====")

        X_train, X_val = X[train_idx], X[val_idx]
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # -----------------------------------------------------
        # Step 5: UMAP
        # -----------------------------------------------------
        print(f"[{now()}] Running UMAP...")
        umap_model = umap.UMAP(
            n_components=3,
            min_dist=0.0,
            n_neighbors=30,
            random_state=42,
            n_jobs=N_JOBS
        )

        X_umap_train = umap_model.fit_transform(X_train)
        X_umap_val = umap_model.transform(X_val)

        # -----------------------------------------------------
        # Step 6: HDBSCAN
        # -----------------------------------------------------
        print(f"[{now()}] Running HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            core_dist_n_jobs=N_JOBS
        )

        train_labels = clusterer.fit_predict(X_umap_train)
        val_labels = np.full(len(X_val), -1)

        # -----------------------------------------------------
        # Save UMAP coordinates
        # -----------------------------------------------------
        df_train["umap1"], df_train["umap2"], df_train["umap3"] = X_umap_train.T
        df_val["umap1"], df_val["umap2"], df_val["umap3"] = X_umap_val.T
        df_train["cluster"], df_val["cluster"] = train_labels, val_labels

        fold_df = pd.concat([df_train, df_val])
        fold_df.to_pickle(os.path.join(OUTPUT_DIR, f"fold{fold+1}_umap_hdbscan_5feat.pkl"))

        df_train.to_csv(os.path.join(OUTPUT_DIR, f"results_fold{fold+1}_train_5feat.csv"), index=False)
        df_val.to_csv(os.path.join(OUTPUT_DIR, f"results_fold{fold+1}_val_5feat.csv"), index=False)

        with open(os.path.join(OUTPUT_DIR, f"umap_model_fold{fold+1}_5feat.pkl"), "wb") as f:
            pickle.dump(umap_model, f)
        with open(os.path.join(OUTPUT_DIR, f"hdbscan_model_fold{fold+1}_5feat.pkl"), "wb") as f:
            pickle.dump(clusterer, f)

        # -----------------------------------------------------
        # Plot (safe on HPC)
        # -----------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.scatter(X_umap_train[:, 0], X_umap_train[:, 1], c=train_labels, s=3)
        plt.title(f"UMAP + HDBSCAN Clusters - Fold {fold+1} (5 features)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fold{fold+1}_clusters_5feat.pdf"))
        plt.close()

        print(f"[{now()}] Fold {fold+1} completed in {round((time.time() - fold_start)/60, 2)} min")

    print(f"\n[{now()}] All folds completed!")
    print(f"[{now()}] Total runtime: {round((time.time() - overall_start)/60, 2)} minutes")
    print(f"[{now()}] All outputs saved under: {OUTPUT_DIR}")


# =============================================================
# Run main()
# =============================================================

if __name__ == "__main__":
    main()