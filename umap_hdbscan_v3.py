#!/usr/bin/env python3
# =============================================================
# UMAP + HDBSCAN semi-supervised anomaly detection for MRD
# Trains on NORMAL reference samples only, scores AML patient
# samples by deviation from the learned normal structure.
# Produces a continuous per-cell anomaly score, consistent with
# the shared evaluation framework (see evaluation_framework.md).
# =============================================================

import os
import glob
import time
import datetime
import pickle

import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fcsparser import parse

# =============================================================
# CONFIGURATION
# =============================================================

N_JOBS = 10
EVENTS_PER_FILE = 5000
FEATURES = ["SSC-A", "CD45 KO", "CD34 Cy55", "CD117"]
N_SPLITS = 5

FCS_FOLDER = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"

# Mapping of sample_id -> group ("normal" or "patient").
# Replace this with however your cohort metadata is actually stored
# (e.g. read from a CSV with columns sample_id, group).
# SAMPLE_GROUPS = pd.read_csv("sample_metadata.csv").set_index("sample_id")["group"].to_dict()
SAMPLE_GROUPS = {}  # <-- fill in: {"sample01": "normal", "sample02": "patient", ...}
NORMAL_LABEL = "normal"
PATIENT_LABEL = "patient"


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
# Step 2: Sample each file to a fixed number of events
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
# Step 3: Attach normal/patient group label per sample
# =============================================================

def attach_group_labels(df, sample_groups, sample_col="sample_id"):
    missing = set(df[sample_col].unique()) - set(sample_groups.keys())
    if missing:
        raise ValueError(
            f"No group label found for {len(missing)} samples, e.g. {list(missing)[:5]}. "
            "Populate SAMPLE_GROUPS before running."
        )
    df = df.copy()
    df["group"] = df[sample_col].map(sample_groups)
    return df


# =============================================================
# Step 4: Per-cell anomaly score for new (val/patient) points
# =============================================================

def score_new_points(clusterer, embedding):
    """
    Returns a per-cell anomaly score (higher = more anomalous /
    more leukemic-like) for points NOT used to fit the clusterer,
    using HDBSCAN's approximate_predict machinery.
    """
    labels, strengths = hdbscan.approximate_predict(clusterer, embedding)

    # approximate_predict_scores gives an outlier-type score comparable
    # to clusterer.outlier_scores_ on the training set, when available.
    try:
        outlier_scores = hdbscan.prediction.approximate_predict_scores(clusterer, embedding)
    except AttributeError:
        # Fallback for older hdbscan versions: invert membership strength
        # so that low confidence in any cluster -> high anomaly score.
        outlier_scores = 1.0 - strengths

    return labels, strengths, outlier_scores


# =============================================================
# MAIN WORKFLOW
# =============================================================

def main():

    # ---------------------------------------------------------
    # Load + prep
    # ---------------------------------------------------------
    df = load_all_fcs(FCS_FOLDER)
    df.to_pickle("blast110_full_dataset.pkl")

    print(f"[{now()}] Sampling events...")
    df = sample_events_per_file(df, n_events=EVENTS_PER_FILE)

    print(f"[{now()}] Cleaning NaNs and infs...")
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=FEATURES)
    print(f"[{now()}] Dropped {before - len(df)} rows")

    df = attach_group_labels(df, SAMPLE_GROUPS)

    df_normal = df[df["group"] == NORMAL_LABEL].reset_index(drop=True)
    df_patient = df[df["group"] == PATIENT_LABEL].reset_index(drop=True)
    print(f"[{now()}] Normal-reference: {df_normal['sample_id'].nunique()} samples, "
          f"{len(df_normal)} events")
    print(f"[{now()}] Patient: {df_patient['sample_id'].nunique()} samples, "
          f"{len(df_patient)} events")

    X_normal = df_normal[FEATURES].values
    groups_normal = df_normal["sample_id"].values

    X_patient = df_patient[FEATURES].values

    # ---------------------------------------------------------
    # GroupKFold over NORMAL samples only.
    # Each fold fits on a subset of normal samples and validates
    # generalization on held-out normal samples (sanity check),
    # while also scoring ALL patient samples against that fold's
    # model (out-of-fold scoring for the actual MRD task).
    # ---------------------------------------------------------
    gkf = GroupKFold(n_splits=N_SPLITS)
    overall_start = time.time()
    patient_fold_scores = []  # collect per-fold patient scores for later averaging

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_normal, groups=groups_normal)):
        fold_start = time.time()
        print(f"\n[{now()}] ===== Fold {fold+1} started =====")

        X_train = X_normal[train_idx]
        df_train = df_normal.iloc[train_idx].copy()

        X_val_normal = X_normal[val_idx]
        df_val_normal = df_normal.iloc[val_idx].copy()

        # -------------------------------------------------
        # UMAP: fit on normal-reference train cells only
        # -------------------------------------------------
        print(f"[{now()}] Fitting UMAP on normal-reference train cells...")
        umap_model = umap.UMAP(
            n_components=3,
            min_dist=0.0,
            n_neighbors=30,
            random_state=42,
            n_jobs=N_JOBS,
        )
        X_umap_train = umap_model.fit_transform(X_train)
        X_umap_val_normal = umap_model.transform(X_val_normal)
        X_umap_patient = umap_model.transform(X_patient)

        # -------------------------------------------------
        # HDBSCAN: fit on normal-reference train embedding,
        # with prediction_data enabled so we can score new points
        # -------------------------------------------------
        print(f"[{now()}] Fitting HDBSCAN on normal-reference train embedding...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            core_dist_n_jobs=N_JOBS,
            prediction_data=True,
        )
        train_labels = clusterer.fit_predict(X_umap_train)
        train_outlier_scores = clusterer.outlier_scores_

        # -------------------------------------------------
        # Score held-out normal cells (sanity check: these
        # should score LOW on anomaly, since they're normal too)
        # -------------------------------------------------
        print(f"[{now()}] Scoring held-out normal validation cells...")
        val_normal_labels, val_normal_strengths, val_normal_scores = score_new_points(
            clusterer, X_umap_val_normal
        )

        # -------------------------------------------------
        # Score ALL patient cells against this fold's model
        # -------------------------------------------------
        print(f"[{now()}] Scoring patient cells...")
        patient_labels, patient_strengths, patient_scores = score_new_points(
            clusterer, X_umap_patient
        )

        # -------------------------------------------------
        # Assemble per-cell outputs
        # -------------------------------------------------
        df_train["umap1"], df_train["umap2"], df_train["umap3"] = X_umap_train.T
        df_train["cluster"] = train_labels
        df_train["anomaly_score"] = train_outlier_scores
        df_train["fold"] = fold + 1
        df_train["split"] = "train_normal"

        df_val_normal["umap1"], df_val_normal["umap2"], df_val_normal["umap3"] = X_umap_val_normal.T
        df_val_normal["cluster"] = val_normal_labels
        df_val_normal["cluster_strength"] = val_normal_strengths
        df_val_normal["anomaly_score"] = val_normal_scores
        df_val_normal["fold"] = fold + 1
        df_val_normal["split"] = "val_normal"

        df_patient_fold = df_patient.copy()
        df_patient_fold["umap1"], df_patient_fold["umap2"], df_patient_fold["umap3"] = X_umap_patient.T
        df_patient_fold["cluster"] = patient_labels
        df_patient_fold["cluster_strength"] = patient_strengths
        df_patient_fold["anomaly_score"] = patient_scores
        df_patient_fold["fold"] = fold + 1
        df_patient_fold["split"] = "patient"

        fold_df = pd.concat([df_train, df_val_normal, df_patient_fold])
        fold_df.to_pickle(f"fold{fold+1}_umap_hdbscan.pkl")
        fold_df.to_csv(f"results_fold{fold+1}.csv", index=False)

        patient_fold_scores.append(
            df_patient_fold[["sample_id", "anomaly_score"]].assign(fold=fold + 1)
        )

        with open(f"umap_model_fold{fold+1}.pkl", "wb") as f:
            pickle.dump(umap_model, f)
        with open(f"hdbscan_model_fold{fold+1}.pkl", "wb") as f:
            pickle.dump(clusterer, f)

        # -------------------------------------------------
        # Plot: train (normal) clusters + patient cells overlaid,
        # colored by anomaly score, so deviation is visible directly
        # -------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.scatter(X_umap_train[:, 0], X_umap_train[:, 1], c="lightgrey", s=3, label="normal (train)")
        sc = plt.scatter(
            X_umap_patient[:, 0], X_umap_patient[:, 1],
            c=patient_scores, cmap="viridis", s=3, label="patient"
        )
        plt.colorbar(sc, label="anomaly score")
        plt.title(f"UMAP + HDBSCAN — Fold {fold+1} (normal reference vs. patient)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(markerscale=4)
        plt.tight_layout()
        plt.savefig(f"fold{fold+1}_clusters.pdf")
        plt.close()

        print(f"[{now()}] Fold {fold+1} completed in {round((time.time() - fold_start)/60, 2)} min")

    # ---------------------------------------------------------
    # Average patient anomaly score across folds (out-of-fold
    # ensemble) for downstream per-cell / per-sample evaluation
    # ---------------------------------------------------------
    all_patient_scores = pd.concat(patient_fold_scores)
    mean_patient_scores = (
        all_patient_scores.groupby(all_patient_scores.index)["anomaly_score"]
        .mean()
        .rename("anomaly_score_mean")
    )
    df_patient_final = df_patient.copy()
    df_patient_final["anomaly_score_mean"] = mean_patient_scores
    df_patient_final.to_csv("patient_anomaly_scores_final.csv", index=False)

    print(f"\n[{now()}] All folds completed!")
    print(f"[{now()}] Total runtime: {round((time.time() - overall_start)/60, 2)} minutes")
    print(f"[{now()}] Final per-cell patient scores saved to patient_anomaly_scores_final.csv")
    print(f"[{now()}] This file is the input to the evaluation framework "
          f"(per-cell AUROC/AUPRC vs. manual LAIP ground truth).")


if __name__ == "__main__":
    main()