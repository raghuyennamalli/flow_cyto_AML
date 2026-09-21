#!/usr/bin/env python3
# hst.py -- End-to-end pipeline: Aggregation (2K/5K), 5% downsample, Pure-NumPy HST, CV, plots
# No river / no scikit-multiflow required.

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
)

# ==================================================
# PATHS & PARAMETERS (edit these if needed)
# ==================================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels"
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/hst_run_v2"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "plots_downsampled"), exist_ok=True)

print(f"[DEBUG] FCS_PATH: {FCS_PATH}")
print(f"[DEBUG] LABEL_PATH: {LABEL_PATH}")
print(f"[DEBUG] OUTPUT_PATH: {OUTPUT_PATH}")
print(f"[DEBUG] FCS files exist: {os.path.exists(FCS_PATH)}")
print(f"[DEBUG] LABEL files exist: {os.path.exists(LABEL_PATH)}")

# Markers/features to use
features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]

# Aggregation thresholds
MIN_EVENTS_PER_SAMPLE = 5000
SAMPLE_2K = 2000
SAMPLE_5K = 5000

# HST hyperparameters (you can tweak)
HST_N_TREES = 50
HST_DEPTH = 12

# CV
OUTER_SPLITS = 3

# ==================================================
# Utility: read FCS -> pandas dataframe
# ==================================================
def from_fcs(path):
    """Read an FCS file using FlowCytometryTools and apply instrument channel mapping."""
    try:
        from FlowCytometryTools import FCMeasurement
    except Exception as e:
        raise ImportError("FlowCytometryTools is required to read .fcs files. Install or load it.") from e

    sample = FCMeasurement(ID='X', datafile=path)

    # Extract meta channel mapping
    try:
        meta = sample.meta['_channels_']
        instrument_cols = meta['$PnN'].tolist()
    except Exception as e:
        print(f"[WARN] Could not extract channel names from meta for {path}, using defaults. Error: {e}")
        df = sample.data.copy()
        if 'event_ID' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'event_ID'})
        return df

    # Load data
    df = sample.data.copy()

    # Rename columns to instrument names
    if len(df.columns) == len(instrument_cols):
        df.columns = instrument_cols
    else:
        print(f"[WARN] Column count mismatch in {path}, keeping original names.")

    # Ensure event_ID exists
    if 'event_ID' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'event_ID'})

    return df


# ==================================================
# Make 5% blast dataset (balanced to 5% blasts)
# ==================================================
def make_5pct_dataset(data, output_path):
    print(f"\n[INFO] Creating 5% blast dataset: {output_path}")
    total_n = len(data)
    blasts = data[data["Blast"] == 1]
    normal = data[data["Blast"] == 0]

    target_blasts = int(total_n * 0.05)
    target_normal = total_n - target_blasts

    if len(blasts) < target_blasts:
        print(f"[ERROR] Not enough blasts: have {len(blasts)}, need {target_blasts}")
        return

    if len(normal) < target_normal:
        print(f"[WARNING] Not enough normal cells, taking all available.")
        target_normal = len(normal)

    blasts_sample = blasts.sample(n=target_blasts, random_state=42)
    normal_sample = normal.sample(n=target_normal, random_state=42)

    df_5pct = pd.concat([blasts_sample, normal_sample], ignore_index=True)
    df_5pct = df_5pct.sample(frac=1, random_state=42).reset_index(drop=True)
    df_5pct.to_pickle(output_path)

    print(f"[SUCCESS] 5% dataset saved: {output_path}")
    print(f"[INFO] Final blast prevalence = {df_5pct['Blast'].mean():.2%}\n")

# ==================================================
# Plotting helpers 
# ==================================================
def plot_confusion_matrix(y_true, y_pred, fold, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['WBC','Blast'], yticklabels=['WBC','Blast'])
    plt.title(f'Confusion Matrix - Fold {fold+1} (HST)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, fold, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold+1} (HST)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, fold, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, lw=2, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold+1} (HST)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_anomaly_score_distribution(y_true, y_scores, fold, output_path):
    blast_scores = y_scores[y_true == 1]
    normal_scores = y_scores[y_true == 0]
    plt.figure(figsize=(10,6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})')
    plt.hist(blast_scores, bins=50, alpha=0.6, label=f'Blast (n={len(blast_scores)})')
    plt.xlabel('Anomaly Score'); plt.ylabel('Frequency')
    plt.title(f'Anomaly Score Distribution - Fold {fold+1}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_anomaly_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_by_class_boxplot(y_true, y_scores, fold, output_path):
    df_scores = pd.DataFrame({'Score': y_scores, 'Class': ['Normal' if y==0 else 'Blast' for y in y_true]})
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df_scores, x='Class', y='Score', palette=['blue','red'])
    plt.title(f'Anomaly Scores by Class - Fold {fold+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_score_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_metrics(threshold_results, output_path):
    results_df = pd.DataFrame(threshold_results)
    if results_df.empty:
        return
    folds = results_df['fold'].values + 1
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    axes = axes.ravel()
    axes[0].bar(folds, results_df['precision']); axes[0].set_title('Precision')
    axes[1].bar(folds, results_df['recall']); axes[1].set_title('Recall')
    axes[2].bar(folds, results_df['f1']); axes[2].set_title('F1')
    means = [results_df['precision'].mean(), results_df['recall'].mean(), results_df['f1'].mean()]
    stds  = [results_df['precision'].std(), results_df['recall'].std(), results_df['f1'].std()]
    axes[3].bar(['P','R','F1'], means, yerr=stds, capsize=6); axes[3].set_title('Avg +/- std')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_performance(all_fold_results, output_path):
    results_df = pd.DataFrame(all_fold_results)
    if results_df.empty:
        print("[WARN] No per-sample results to plot.")
        return
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    sample_recall = results_df.groupby('sample_id')['recall'].mean().sort_values()
    axes[0,0].barh(sample_recall.index, sample_recall.values); axes[0,0].set_title('Recall per sample')
    sample_precision = results_df.groupby('sample_id')['precision'].mean().sort_values()
    axes[0,1].barh(sample_precision.index, sample_precision.values); axes[0,1].set_title('Precision per sample')
    sample_f1 = results_df.groupby('sample_id')['f1'].mean().sort_values()
    axes[1,0].barh(sample_f1.index, sample_f1.values); axes[1,0].set_title('F1 per sample')
    gt_counts = results_df.groupby('sample_id')['gt_count'].first()
    pred_counts = results_df.groupby('sample_id')['pred_count'].first()
    x = np.arange(len(gt_counts)); width=0.35
    axes[1,1].bar(x-width/2, gt_counts.values, width, label='GT'); axes[1,1].bar(x+width/2, pred_counts.values, width, label='Pred')
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(gt_counts.index, rotation=90)
    axes[1,1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sample_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ==================================================
# Threshold finding 
# ==================================================
def find_optimal_threshold(y_true, y_scores, method='f1'):
    """Return (threshold, metrics dict)"""
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        optimal_thresh = thresholds[ix]
    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        fscore = (2 * precision * recall) / (precision + recall + 1e-10)
        ix = np.argmax(fscore)
        # thresholds is shorter than precision/recall by 1
        if ix < len(thresholds):
            optimal_thresh = thresholds[ix]
        else:
            optimal_thresh = thresholds[-1] if len(thresholds)>0 else 0.0
    elif method == 'clinical_recall':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        thresholds = np.append(0, thresholds)
        valid_idx = precision >= 0.90
        if np.sum(valid_idx) > 0:
            valid_recall = recall[valid_idx]
            valid_thresholds = thresholds[valid_idx]
            ix = np.argmax(valid_recall)
            optimal_thresh = valid_thresholds[ix]
        else:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            fscore = (2 * precision * recall) / (precision + recall + 1e-10)
            ix = np.argmax(fscore)
            if ix < len(thresholds):
                optimal_thresh = thresholds[ix]
            else:
                optimal_thresh = thresholds[-1] if len(thresholds)>0 else 0.0
    else:
        raise ValueError("Unknown method")

    y_pred = (y_scores >= optimal_thresh).astype(int)
    metrics = {
        'threshold': optimal_thresh,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return optimal_thresh, metrics

# ==================================================
# PURE NUMPY HST implementation (robust enough)
# ==================================================
import numpy as np

class HalfSpaceTree:
    """
    Simple offline Half-Space Tree style anomaly detector.

    - Build random half-space trees over the min-max range of the training data.
    - During fit, only NORMAL points (y==0) should be passed, so nodes store "mass" of normal data.
    - During scoring, nodes with low mass and shallow depth contribute higher anomaly.

    This is an HST-inspired, offline approximation (no sliding window).
    """

    def __init__(self, depth=12, n_trees=50, random_state=None):
        self.depth = int(depth)
        self.n_trees = int(n_trees)
        self.rng = np.random.default_rng(random_state)
        self.trees = []  # each tree is a dict root

    # ---------- tree construction ----------

    def _make_tree(self, mins, maxs, n_features):
        """Recursively sample random half-space splits; represent as nested dicts."""
        def build(d, mins_loc, maxs_loc):
            # leaf node
            if d == 0:
                return {
                    "feat": None,
                    "thresh": None,
                    "left": None,
                    "right": None,
                    "mass": 0.0,    # number of normal points that reached this node
                    "depth": 0      # will be filled later
                }

            feat = int(self.rng.integers(0, n_features))
            if maxs_loc[feat] == mins_loc[feat]:
                thresh = float(mins_loc[feat])
            else:
                thresh = float(self.rng.uniform(mins_loc[feat], maxs_loc[feat]))

            mins_l, maxs_l = mins_loc.copy(), maxs_loc.copy()
            maxs_l[feat] = thresh
            mins_r, maxs_r = mins_loc.copy(), maxs_loc.copy()
            mins_r[feat] = thresh

            node = {
                "feat": feat,
                "thresh": thresh,
                "left": build(d - 1, mins_l, maxs_l),
                "right": build(d - 1, mins_r, maxs_r),
                "mass": 0.0,   # internal nodes also track mass
                "depth": 0     # will be updated after construction
            }
            return node

        root = build(self.depth, mins.copy(), maxs.copy())
        self._set_depths(root, depth=0)
        return root

    def _set_depths(self, node, depth):
        if node is None:
            return
        node["depth"] = depth
        self._set_depths(node["left"], depth + 1)
        self._set_depths(node["right"], depth + 1)

    # ---------- fitting: accumulate mass from normal points ----------

    def fit(self, X):
        """
        Fit on a 2D array X of NORMAL points only (y == 0).

        If you pass mixed data, the "mass" will include blasts too,
        which weakens the anomaly concept.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        n_features = X.shape[1]

        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        # build trees
        self.trees = [self._make_tree(mins, maxs, n_features) for _ in range(self.n_trees)]

        # accumulate node mass for each normal point
        for x in X:
            for tree in self.trees:
                self._update_mass_single(tree, x)

        return self

    def _update_mass_single(self, node, x):
        # increment mass at this node, then go down
        node["mass"] += 1.0
        if node["feat"] is None:
            return  # leaf
        feat = node["feat"]
        if x[feat] <= node["thresh"]:
            self._update_mass_single(node["left"], x)
        else:
            self._update_mass_single(node["right"], x)

    # ---------- scoring ----------

    def _score_one_tree(self, node, x, alpha=1e-3, beta=1.0):
        """
        Recursive contribution of one tree to anomaly score.

        - alpha: smoothing for mass (avoid division by zero)
        - beta: weight on depth (shallower + low mass => more anomalous)

        Higher return value = more anomalous.
        """
        if node is None:
            return 0.0

        # contribution at this node
        m = node["mass"]
        d = node["depth"]
        # small mass and small depth => high anomaly
        contrib = 1.0 / (alpha + m) * (1.0 + beta * (1.0 / (1 + d)))

        if node["feat"] is None:
            return contrib

        feat = node["feat"]
        if x[feat] <= node["thresh"]:
            return contrib + self._score_one_tree(node["left"], x, alpha=alpha, beta=beta)
        else:
            return contrib + self._score_one_tree(node["right"], x, alpha=alpha, beta=beta)

    def score_samples(self, X):
        """
        Return anomaly scores for X: higher score = more anomalous.
        """
        X = np.asarray(X)
        out = np.zeros(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            scores = [self._score_one_tree(t, x) for t in self.trees]
            out[i] = np.mean(scores)
        return out

    def decision_function(self, X):
        return self.score_samples(X)


class HSTWrapper:
    """
    Simple wrapper with the same API you used for Isolation Forest and LOF.
    """

    def __init__(self, n_trees=50, depth=12, random_state=None):
        self.model = HalfSpaceTree(depth=depth, n_trees=n_trees, random_state=random_state)

    def fit(self, X):
        # IMPORTANT: pass only normal (WBC) cells if you want one-class behaviour
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)

# ==================================================
# STEP 1: CREATE AGGREGATED DATASET (2K + 5K)
# ==================================================
pkl_2k = os.path.join(OUTPUT_PATH, "BLAST110_2K.pkl")
pkl_5k = os.path.join(OUTPUT_PATH, "BLAST110_5K.pkl")

if not os.path.exists(pkl_5k):
    print("[INFO] Building 2K/5K aggregated datasets from FCS files.")
    dataframes_2K = []
    dataframes_5K = []

    for root, dirs, files in os.walk(FCS_PATH):
        for file in files:
            if not file.endswith(".fcs"):
                continue

            name = file.replace(".fcs", "")
            parts = name.split("_")
            if len(parts) < 3:
                print(f"[WARN] Unexpected filename format: {file}, skipping.")
                continue
            patient_id = "_".join(parts[:2])
            sample_id = parts[2]

            fcs_path = os.path.join(root, file)
            label_file = os.path.join(LABEL_PATH, f"{patient_id}_{sample_id}.csv")

            if not os.path.exists(label_file):
                print(f"[WARN] Label missing: {label_file}, skipping.")
                continue

            try:
                ff = from_fcs(fcs_path)
            except Exception as e:
                print(f"[WARN] Failed to read {fcs_path}: {e}")
                continue

            labels = pd.read_csv(label_file, index_col=0)
            # Merge on event_ID
            if 'event_ID' not in ff.columns and labels.index.name == 'event_ID':
                ff = ff.reset_index().rename(columns={'index': 'event_ID'})

            merged = pd.merge(ff, labels, on="event_ID", how="inner")

            # Check columns
            expected_cols = features + ["Blast", "WBC", "Singlets", "event_ID"]
            missing = [c for c in expected_cols if c not in merged.columns]
            if missing:
                print(f"[WARN] Missing columns {missing} in {file}, skipping.")
                continue

            merged = merged[features + ["Blast", "WBC", "Singlets", "event_ID"]]
            merged = merged[(merged["Singlets"] == 1) & (merged["WBC"] == 1)]
            merged = merged.drop(columns=["Singlets", "WBC"])
            merged["patient_id"] = patient_id
            merged["sample_id"] = sample_id

            if len(merged) >= MIN_EVENTS_PER_SAMPLE:
                dataframes_2K.append(merged.sample(n=SAMPLE_2K, random_state=42))
                dataframes_5K.append(merged.sample(n=SAMPLE_5K, random_state=42))
            else:
                print(f"[INFO] Sample {file} has {len(merged)} events (<{MIN_EVENTS_PER_SAMPLE}), skipping.")

    if len(dataframes_2K) == 0 or len(dataframes_5K) == 0:
        raise RuntimeError("[ERROR] No valid samples were found to build 2K/5K datasets. Check paths and files.")

    data_2K = pd.concat(dataframes_2K).reset_index(drop=True)
    data_5K = pd.concat(dataframes_5K).reset_index(drop=True)

    data_2K.to_pickle(pkl_2k)
    data_5K.to_pickle(pkl_5k)
    print(f"[INFO] Saved {pkl_2k} and {pkl_5k}")

    # Create 5% blast datasets
    make_5pct_dataset(data=data_2K, output_path=os.path.join(OUTPUT_PATH, "BLAST110_2K_5pct.pkl"))
    make_5pct_dataset(data=data_5K, output_path=os.path.join(OUTPUT_PATH, "BLAST110_5K_5pct.pkl"))
else:
    print("[INFO] Pickled aggregated data already exists. Skipping aggregation step.")

# ==================================================
# STEP 2: LOAD DOWN-SAMPLED 5K 5% DATASET
# ==================================================
pkl_downsampled = os.path.join(OUTPUT_PATH, "BLAST110_5K_5pct.pkl")
if not os.path.exists(pkl_downsampled):
    raise FileNotFoundError(f"{pkl_downsampled} not found. Aggregation step should create it.")
data = pd.read_pickle(pkl_downsampled)

X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]

X_scaled = X.values
print(f"[INFO] Data shape: {X_scaled.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ==================================================
# STEP 3: OUTER CV (GroupKFold) and HST evaluation
# ==================================================
outer_cv = GroupKFold(n_splits=OUTER_SPLITS)

total_start_time = time.time()
threshold_results = []
all_fold_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y, groups)):
    print("\n" + "="*70)
    print(f"OUTER FOLD {fold+1}/{outer_cv.get_n_splits()}")
    print("="*70)
    fold_start = time.time()

    X_train = X_scaled[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_scaled[test_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train blast %: {y_train.mean():.2%}, Test blast %: {y_test.mean():.2%}")

    # Option: train only on normals (uncomment to enable one-class training)
    X_train_fit = X_train[y_train.values == 0]
    

    # Train HST
    model = HSTWrapper(n_trees=HST_N_TREES, depth=HST_DEPTH, random_state=fold)
    model.fit(X_train_fit)

    # Score test set
    anomaly_scores_test = model.decision_function(X_test)

    # Select optimal threshold via F1 on this fold
    optimal_thresh, metrics = find_optimal_threshold(y_test.values, anomaly_scores_test, method='f1')
    preds = (anomaly_scores_test >= optimal_thresh).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"\n[OUTER CV] Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(f"\n[OUTER CV] Classification Report:")
    print(classification_report(y_test, preds, target_names=["WBC","Blast"]))
    print(f"\n[THRESHOLD ANALYSIS] Optimal threshold: {optimal_thresh:.6f}")
    print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")

    threshold_results.append({
        'fold': fold,
        'optimal_threshold': optimal_thresh,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'train_size': X_train_fit.shape[0]
    })

    # ==== Visualizations for this fold ====
    plot_confusion_matrix(y_test, preds, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_roc_curve(y_test, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_precision_recall_curve(y_test, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_anomaly_score_distribution(y_test.values, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_score_by_class_boxplot(y_test.values, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))

    # ==== Per-sample aggregation & metrics ====
    per_sample_start = time.time()
    sample_ids = samples.iloc[test_idx].unique()
    model_results = []

    for sid in sample_ids:
        subset = data[data["sample_id"] == sid]
        X_sub = subset[features].values
        y_sub = subset["Blast"].values
        scores_sub = model.decision_function(X_sub)
        pred_sub = (scores_sub >= optimal_thresh).astype(int)

        result_dict = {
            "fold": fold,
            "sample_id": sid,
            "model": "HST_numpy",
            "gt_count": int(np.sum(y_sub == 1)),
            "gt_perc": float(np.mean(y_sub == 1)),
            "pred_count": int(np.sum(pred_sub == 1)),
            "pred_perc": float(np.mean(pred_sub == 1)),
            "accuracy": accuracy_score(y_sub, pred_sub),
            "precision": precision_score(y_sub, pred_sub, zero_division=0),
            "recall": recall_score(y_sub, pred_sub, zero_division=0),
            "f1": f1_score(y_sub, pred_sub, zero_division=0),
            "mean_anomaly_score_blast": np.mean(scores_sub[y_sub == 1]) if np.any(y_sub == 1) else np.nan,
            "mean_anomaly_score_wbc": np.mean(scores_sub[y_sub == 0]) if np.any(y_sub == 0) else np.nan,
            "optimal_threshold": optimal_thresh
        }
        model_results.append(result_dict)

    per_sample_elapsed = time.time() - per_sample_start
    pd.DataFrame(model_results).to_csv(os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_downsampled_outerCV.csv"), index=False)
    all_fold_results = all_fold_results + model_results

    fold_elapsed = time.time() - fold_start
    print(f"\n[FOLD TIMING] Fold time: {fold_elapsed:.1f}s")

# ==================================================
# SAVE SUMMARIES + FINAL PLOTS
# ==================================================
pd.DataFrame(threshold_results).to_csv(os.path.join(OUTPUT_PATH, "CV", "threshold_summary.csv"), index=False)
pd.DataFrame(all_fold_results).to_csv(os.path.join(OUTPUT_PATH, "CV", "all_samples_results.csv"), index=False)

print("[VISUALIZATION] Creating summary plots...")
plot_summary_metrics(threshold_results, os.path.join(OUTPUT_PATH, "plots_downsampled"))
plot_sample_performance(all_fold_results, os.path.join(OUTPUT_PATH, "plots_downsampled"))

total_elapsed = time.time() - total_start_time
print("\n=== NESTED CV COMPLETE ===")
print(f"Total runtime: {total_elapsed:.1f}s = {total_elapsed/60:.1f}min")

print(f"[OUTPUT FILES] {os.path.join(OUTPUT_PATH, 'CV')}")
