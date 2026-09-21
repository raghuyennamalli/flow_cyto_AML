import os
import time
import numpy as np
import pandas as pd
from FlowCytometryTools import FCMeasurement
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer,
    precision_recall_curve, roc_curve
)
from joblib import parallel_backend



# ==========================================
# PARAMETERS & PATHS
# ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)

features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]
# ==========================================
# HELPER FUNCTIONS
# ==========================================

def from_fcs(path):
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()

# Custom scorer for unsupervised isolation forest (uses score_samples; higher = better for normality)
def iforest_norm_score(estimator, X):
    try:
        if not hasattr(estimator, 'score_samples'):
            return 0  # Handle corruption gracefully
        return np.mean(estimator.score_samples(X))
    except AttributeError:
        return 0

def find_optimal_threshold(y_true, y_scores, method='clinical_recall'):
    """
    Find optimal threshold for Isolation Forest decision function.
    
    Isolation Forest convention:
    - Higher scores = more normal
    - Lower scores = more anomalous
    
    We want: scores >= threshold ? classify as blast (class 1)
    """
    
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        optimal_thresh = thresholds[ix]
        
    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        fscore = (2 * precision * recall) / (precision + recall + 1e-10)
        ix = np.argmax(fscore)
        optimal_thresh = thresholds[ix] if ix < len(thresholds) else thresholds[-1]
        
    elif method == 'clinical_recall':
        # Maximize recall while maintaining >= 90% precision
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        thresholds = np.append(0, thresholds) 
        valid_idx = precision >= 0.90
        if np.sum(valid_idx) > 0:
            valid_recall = recall[valid_idx]
            valid_thresholds = thresholds[valid_idx]
            ix = np.argmax(valid_recall)
            optimal_thresh = valid_thresholds[ix]
        else:
            fscore = (2 * precision * recall) / (precision + recall + 1e-10)
            ix = np.argmax(fscore)
            optimal_thresh = thresholds[ix] if ix < len(thresholds) else thresholds[-1]
    
    # Apply threshold
    y_pred = (y_scores >= optimal_thresh).astype(int)
    
    metrics = {
        'threshold': optimal_thresh,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return optimal_thresh, metrics

# ==========================================
# STEP 1: CREATE AGGREGATED DATASET (2K + 5K)
# ==========================================
pkl_2k = os.path.join(OUTPUT_PATH, "BLAST110_2K.pkl")
pkl_5k = os.path.join(OUTPUT_PATH, "BLAST110_5K.pkl")

if not os.path.exists(pkl_5k):
    dataframes_2K = []
    dataframes_5K = []
    for root, dirs, files in os.walk(FCS_PATH):
        for file in files:
            if not file.endswith(".fcs"):
                continue
            name = file.replace(".fcs", "")
            parts = name.split("_")
            patient_id = "_".join(parts[:2])
            sample_id = parts[2] 
            fcs_path = os.path.join(root, file)
            label_file = os.path.join(LABEL_PATH, f"{patient_id}_{sample_id}.csv")
            if not os.path.exists(label_file):
                print(f" {label_file} - Label file missing, skipping.")
                continue

            ff = from_fcs(fcs_path)
            labels = pd.read_csv(label_file, index_col=0)
            ff = pd.merge(ff, labels, on="event_ID")
            ff = ff[features + ["Blast", "event_ID"]]
            ff["patient_id"] = patient_id
            ff["sample_id"] = sample_id

            # Sample 2K and 5K events per FCS
            if len(ff) >= 5000:
                dataframes_2K.append(ff.sample(n=2000, random_state=42))
                dataframes_5K.append(ff.sample(n=5000, random_state=42))

    data_2K = pd.concat(dataframes_2K).reset_index(drop=True)
    data_5K = pd.concat(dataframes_5K).reset_index(drop=True)
    data_2K.to_pickle(pkl_2k)
    data_5K.to_pickle(pkl_5k)
else:
    print("[INFO] Pickled data already exists.")
    
# ==========================================
# STEP 1: LOAD AGGREGATED DATA
# ==========================================
data = pd.read_pickle(pkl_5k)

X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]

# Scale

X_scaled= X.values

print(f"[INFO] Data shape: {X_scaled.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ==========================================
# STEP 2: DEFINE INNER + OUTER CV
# ==========================================
outer_cv = GroupKFold(n_splits=3)
inner_cv = GroupKFold(n_splits=3)

# Define parameter grid for Isolation Forest

# ==========================================
# STEP 3: OUTER CV LOOP  (WITH PARALLELIZATION)
# ==========================================
total_start_time = time.time()
threshold_results = []
all_fold_results = []
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y, groups)):
    print(f"\n{'='*70}")
    print(f"OUTER FOLD {fold+1}/{outer_cv.get_n_splits()}")
    print(f"{'='*70}")
    
    fold_start_time = time.time()

    X_train = X_scaled[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_scaled[test_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]
    X_train_all = X_train
    train_groups_all = train_groups
    blast_prevalence = y_train.mean()
    print(f"Train size: {X_train_all.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")
    print(f"Train blast %: {y_train.mean():.2%}, Test blast %: {y_test.mean():.2%}")
    # Train only on normal (WBC)
    
    contamination_range = [
    blast_prevalence - 0.03,
    blast_prevalence - 0.02,
    blast_prevalence - 0.01,
    blast_prevalence,
    blast_prevalence + 0.01,
    blast_prevalence + 0.02,
    blast_prevalence + 0.03]
    contamination_range = [max(0.001, c) for c in contamination_range]
    param_grid = {
    "n_estimators": [100, 200, 300],
    "max_samples": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.75, 1.0],
    'contamination':  contamination_range }
    print(f"[INFO] Contamination range: {[f'{c:.4f}' for c in contamination_range]}")

    print(f"Training on {X_train_all.shape[0]} normal cells")

    # ------------------------
    # INNER CV (Grid Search)
    # ------------------------
    print(f"\n[INNER CV] Starting grid search with parallelization...")
    base_iso = IsolationForest(
        random_state=fold,
        n_jobs=-1  # Parallelize tree building
    )
    inner_cv_start = time.time() # Adjust based on available cores
    grid = GridSearchCV(
            base_iso,
            param_grid,
            cv=inner_cv,
            scoring=make_scorer(iforest_norm_score, greater_is_better=True),
            n_jobs=4,  # ? Parallel hyperparameter search (safe with threading)
            refit=True,
            verbose=0)

    # Create fake labels for inner CV (Isolation Forest is unsupervised)
    
    start = time.time()
    grid.fit(X_train_all, y_train, groups=train_groups_all)
    elapsed_inner = time.time() - start
    inner_cv_elapsed = time.time() - inner_cv_start

    # Save inner CV results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_IsolationForest_innerCV.csv"), index=False)
    
    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best Score: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")
    # ------------------------
    # OUTER TESTING
    # ------------------------
    outer_test_start = time.time()
    
    best_iso = grid.best_estimator_

    anomaly_scores_test = best_iso.decision_function(X_test)
    optimal_thresh_clinical, metrics = find_optimal_threshold(y_test.values, anomaly_scores_test, method='clinical_recall')
    preds = (anomaly_scores_test >= optimal_thresh_clinical).astype(int)


    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)  
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    outer_test_elapsed = time.time() - outer_test_start

    print(f"\n[OUTER CV] Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(f"\n[OUTER CV] Classification Report:")
    print(classification_report(y_test, preds, target_names=["WBC", "Blast"]))
    
    print(f"\n[THRESHOLD ANALYSIS]")
    print(f"Optimal threshold: {optimal_thresh_clinical:.4f}")
    print(f"  ? Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")

    threshold_results.append({
        'fold': fold,
        'best_params': str(grid.best_params_),
        'optimal_threshold': optimal_thresh_clinical,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'runtime_seconds': elapsed_inner,
        'inner_cv_time': inner_cv_elapsed,
        'outer_test_time': outer_test_elapsed
    })

    # ------------------------
    # PER-SAMPLE RESULTS
    # ------------------------
    per_sample_start = time.time()
    sample_ids = samples.iloc[test_idx].unique()
    model_results = []

    for sid in sample_ids:
        subset = data[data["sample_id"] == sid]
        X_sub = subset[features].values
        y_sub = subset["Blast"].values
        scores_sub = best_iso.decision_function(X_sub)
        pred_sub = (scores_sub >= optimal_thresh_clinical).astype(int)

        result_dict = {
            "fold": fold,
            "sample_id": sid,
            "model": "IsolationForest",
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
            "optimal_threshold": optimal_thresh_clinical
        }
        model_results.append(result_dict)

    # Save outer CV per-sample results
    per_sample_elapsed = time.time() - per_sample_start
    pd.DataFrame(model_results).to_csv(
        os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_IsolationForest_outerCV.csv"),
        index=False
    )
    all_fold_results.extend(model_results)
    fold_elapsed = time.time() - fold_start_time
    
    print(f"\n[FOLD TIMING]")
    print(f"  - Inner CV: {inner_cv_elapsed:.1f}s")
    print(f"  - Outer test: {outer_test_elapsed:.1f}s")
    print(f"  - Per-sample: {per_sample_elapsed:.1f}s")
    print(f"  - Total fold time: {fold_elapsed:.1f}s")
    
    threshold_results[-1]['total_fold_time'] = fold_elapsed
    threshold_results[-1]['per_sample_time'] = per_sample_elapsed

print("\n=== Nested Cross-Validation Complete ===")

# ==========================================
# SAVE FINAL SUMMARIES
# ==========================================
pd.DataFrame(threshold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "threshold_summary.csv"),
    index=False
)

pd.DataFrame(all_fold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "all_samples_results.csv"),
    index=False
)

# ==========================================
# FINAL RUNTIME REPORT
# ==========================================
total_elapsed = time.time() - total_start_time

print("\n" + "="*70)
print("NESTED CROSS-VALIDATION COMPLETE")
print("="*70)

# ? ANSWER #3: Print total runtime
print(f"\n[RUNTIME SUMMARY]")
print(f"Total runtime: {total_elapsed:.1f} seconds")
print(f"             = {total_elapsed/60:.1f} minutes")
print(f"             = {total_elapsed/3600:.2f} hours")

# Detailed breakdown
avg_results = pd.DataFrame(threshold_results)
print(f"\n[PER-FOLD BREAKDOWN]")
print(f"Average total fold time: {avg_results['total_fold_time'].mean():.1f}s")
print(f"  - Inner CV: {avg_results['inner_cv_time'].mean():.1f}s")
print(f"  - Outer test: {avg_results['outer_test_time'].mean():.1f}s")
print(f"  - Per-sample: {avg_results['per_sample_time'].mean():.1f}s")

print(f"\n[PERFORMANCE SUMMARY]")
print(f"Average Precision: {avg_results['precision'].mean():.3f} + or - {avg_results['precision'].std():.3f}")
print(f"Average Recall: {avg_results['recall'].mean():.3f}  +/-{avg_results['recall'].std():.3f}")
print(f"Average F1: {avg_results['f1'].mean():.3f} +\- {avg_results['f1'].std():.3f}")

print(f"\n[OUTPUT FILES]")
print(f"Threshold summary: {os.path.join(OUTPUT_PATH, 'CV', 'threshold_summary.csv')}")
print(f"Sample results: {os.path.join(OUTPUT_PATH, 'CV', 'all_samples_results.csv')}")

