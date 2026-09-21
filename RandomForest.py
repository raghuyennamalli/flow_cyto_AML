import os
import time
import numpy as np
import pandas as pd
from FlowCytometryTools import FCMeasurement
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer,
    precision_recall_curve, roc_curve
)
from joblib import parallel_backend

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# PARAMETERS & PATHS
# ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output_RF"
EXTERNAL_FCS_PATH = "/storage/mezya.sezen/mphasis/LAIP29/scaled/"
EXTERNAL_LABEL_PATH = "/storage/mezya.sezen/mphasis/LAIP29/labels/"
external_fcs_files = [
    "LAIP29_9_FU_P3_Cleaned_transformed_scaled.fcs", 
    "LAIP29_8_Dx_P2_Cleaned_transformed_scaled.fcs" # <- Add your external files here
]
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "Plots"), exist_ok=True)


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

def evaluate_external_fcs(fcs_path, label_path, best_estimator, optimal_threshold, fold, features):
    """
    Evaluate trained model on external FCS file.
    Returns per-sample metrics if labels available, else predictions only.
    """
    try:
        ff = from_fcs(fcs_path)
        fcs_name = os.path.basename(fcs_path).replace('.fcs', '')
        
        # Check if labels exist
        has_labels = False
        if os.path.exists(label_path):
            labels = pd.read_csv(label_path, index_col=0)
            ff = pd.merge(ff, labels, on="event_ID", how="left")
            has_labels = "Blast" in ff.columns
        
        # Extract features
        if not all(feat in ff.columns for feat in features):
            print(f"  [WARNING] {fcs_name}: Missing features, skipping.")
            return None
        
        X_ext = ff[features].values
        
        # Predictions
        proba_ext = best_estimator.predict_proba(X_ext)[:, 1]
        pred_ext = (proba_ext >= optimal_threshold).astype(int)
        
        result = {
            'fold': fold,
            'external_file': fcs_name,
            'n_cells': len(X_ext),
            'n_predicted_blasts': int(np.sum(pred_ext == 1)),
            'pred_blast_perc': float(np.mean(pred_ext == 1)),
            'mean_proba_predicted_blasts': np.mean(proba_ext[pred_ext == 1]) if np.sum(pred_ext == 1) > 0 else np.nan,
            'mean_proba_predicted_wbc': np.mean(proba_ext[pred_ext == 0]) if np.sum(pred_ext == 0) > 0 else np.nan,
            'optimal_threshold': optimal_threshold,
            'has_labels': has_labels
        }
        
        # If labels available, compute metrics
        if has_labels:
            y_ext = ff["Blast"].values
            result.update({
                'gt_count': int(np.sum(y_ext == 1)),
                'gt_perc': float(np.mean(y_ext == 1)),
                'accuracy': accuracy_score(y_ext, pred_ext),
                'precision': precision_score(y_ext, pred_ext, zero_division=0),
                'recall': recall_score(y_ext, pred_ext, zero_division=0),
                'f1': f1_score(y_ext, pred_ext, zero_division=0),
                'mean_proba_blast': np.mean(proba_ext[y_ext == 1]) if np.any(y_ext == 1) else np.nan,
                'mean_proba_wbc': np.mean(proba_ext[y_ext == 0]) if np.any(y_ext == 0) else np.nan,
            })
        
        return result
    
    except Exception as e:
        print(f"  [ERROR] {fcs_path}: {str(e)}")
        return None

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
                print(f"{label_file} - Label file missing, skipping.")
                continue

            ff = from_fcs(fcs_path)
            labels = pd.read_csv(label_file, index_col=0)
            ff = pd.merge(ff, labels, on="event_ID")

            ff = ff[features + ["Blast", "WBC", "Singlets", "event_ID"]]

            ff = ff[(ff["Singlets"] == 1) & (ff["WBC"] == 1)]
            ff = ff.drop(columns=["Singlets", "WBC"])

            ff["patient_id"] = patient_id
            ff["sample_id"] = sample_id

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
from sklearn.model_selection import GroupShuffleSplit
outer_cv = GroupShuffleSplit(n_splits=5, test_size=0.4, random_state=42)
inner_cv = GroupKFold(n_splits=5)

# Define parameter grid for Isolation Forest

param_grid = {
    "max_depth": range(2, 11),
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 4, 8],
    "n_estimators": [100, 200, 300]
}


# ==========================================
# STEP 3: OUTER CV LOOP  (WITH PARALLELIZATION)
# ==========================================
total_start_time = time.time()
threshold_results = []
all_fold_results = []
external_results = []
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
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")
    print(f"Train blast %: {y_train.mean():.2%}, Test blast %: {y_test.mean():.2%}")
    # Train only on normal (WBC)
    X_train_all = X_train
    train_groups_all = train_groups
    print(f"Training on {X_train.shape[0]} ")

    # ------------------------
    # INNER CV (Grid Search)
    # ------------------------
    print(f"\n[INNER CV] Starting grid search with parallelization...")
    base_rf = RandomForestClassifier(random_state=fold, n_jobs=-1)
    inner_cv_start = time.time() # Adjust based on available cores
    grid = GridSearchCV(
            base_rf,
            param_grid,
            cv=inner_cv,
            scoring='f1',
            n_jobs=4,  # ? Parallel hyperparameter search (safe with threading)
            refit=True,
            verbose=0)

    
    start = time.time()
    grid.fit(X_train, y_train, groups=train_groups)
    elapsed_inner = time.time() - start
    inner_cv_elapsed = time.time() - inner_cv_start

    # Save inner CV results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_RandomForest_innerCV.csv"), index=False)
    
    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best Score: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")
    # ------------------------
    # OUTER TESTING
    # ------------------------
    outer_test_start = time.time()
    
    best_rf = grid.best_estimator_

    proba_scores_test = best_rf.predict_proba(X_test)[:, 1]
    optimal_thresh_clinical, metrics = find_optimal_threshold(y_test.values, proba_scores_test, method='f1')
    preds = (proba_scores_test >= optimal_thresh_clinical).astype(int)


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
    import pickle
    # Save predictions for later threshold analysis
    predictions_data = {
    'fold': fold,
    'y_true': y_test.values,
    'proba_scores': proba_scores_test,
    'optimal_threshold': optimal_thresh_clinical}
    pred_path = os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_predictions.pkl")
    with open(pred_path, 'wb') as f:
       pickle.dump(predictions_data, f)

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
        proba_sub = best_rf.predict_proba(X_sub)[:, 1]
        pred_sub = (proba_sub >= optimal_thresh_clinical).astype(int)

        result_dict = {
            "fold": fold,
            "sample_id": sid,
            "model": "RandomForest",
            "gt_count": int(np.sum(y_sub == 1)),
            "gt_perc": float(np.mean(y_sub == 1)),
            "pred_count": int(np.sum(pred_sub == 1)),
            "pred_perc": float(np.mean(pred_sub == 1)),
            "accuracy": accuracy_score(y_sub, pred_sub),
            "precision": precision_score(y_sub, pred_sub, zero_division=0),
            "recall": recall_score(y_sub, pred_sub, zero_division=0),
            "f1": f1_score(y_sub, pred_sub, zero_division=0),
            "mean_proba_blast": np.mean(proba_sub[y_sub == 1]) if np.any(y_sub == 1) else np.nan,
            "mean_proba_wbc": np.mean(proba_sub[y_sub == 0]) if np.any(y_sub == 0) else np.nan,
            "optimal_threshold": optimal_thresh_clinical
        }
        model_results.append(result_dict)

    # Save outer CV per-sample results
    per_sample_elapsed = time.time() - per_sample_start
    pd.DataFrame(model_results).to_csv(
        os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_RandomForest_outerCV.csv"),
        index=False
    )
    all_fold_results.extend(model_results)
    # ========================================================================================
    # EXTERNAL FCS EVALUATION (NEW SECTION - ADDED AFTER INTERNAL TESTING)
    # ========================================================================================
    print(f"\n{'='*70}")
    print(f"[EXTERNAL EVALUATION] Evaluating on external FCS files...")
    print(f"{'='*70}")
    
    external_eval_start = time.time()
    
    
    for ext_file in external_fcs_files:
        ext_fcs_path = os.path.join(EXTERNAL_FCS_PATH, ext_file)
        # Extract patient ID and sample ID for label matching
        base_name = os.path.basename(ext_file).replace("_Cleaned_transformed_scaled.fcs", "")
        # Example: LAIP29_9_FU_P3_Cleaned_transformed_scaled.fcs ? LAIP29_9_FU_P3
        label_path = os.path.join(EXTERNAL_LABEL_PATH, f"{base_name}.csv")

        
        print(f"\n  Evaluating: {ext_file}")
        
        ext_result = evaluate_external_fcs(
            fcs_path=ext_fcs_path,
            label_path=label_path,
            best_estimator=best_rf,
            optimal_threshold=optimal_thresh_clinical,
            fold=fold,
            features=features
        )
        
        if ext_result is not None:
            external_results.append(ext_result)
            
            # Print results
            print(f"    - Total cells: {ext_result['n_cells']}")
            print(f"    - Predicted blasts: {ext_result['n_predicted_blasts']} ({ext_result['pred_blast_perc']:.2%})")
            print(f"    - Mean proba (blasts): {ext_result['mean_proba_predicted_blasts']:.4f}")
            print(f"    - Mean proba (WBC): {ext_result['mean_proba_predicted_wbc']:.4f}")
            
            if ext_result['has_labels']:
                print(f"    - Ground truth blasts: {ext_result['gt_count']} ({ext_result['gt_perc']:.2%})")
                print(f"    - Accuracy: {ext_result['accuracy']:.4f}, Precision: {ext_result['precision']:.4f}, Recall: {ext_result['recall']:.4f}, F1: {ext_result['f1']:.4f}")
            else:
                print(f"    - [WARNING] No labels found for this file (predictions only)")
    
    external_eval_elapsed = time.time() - external_eval_start

    # ========================================================================================
    # END EXTERNAL EVALUATION
    # ========================================================================================
    
    fold_elapsed = time.time() - fold_start_time
    
    print(f"\n[FOLD TIMING]")
    print(f"  - Inner CV: {inner_cv_elapsed:.1f}s")
    print(f"  - Outer test: {outer_test_elapsed:.1f}s")
    print(f"  - Per-sample: {per_sample_elapsed:.1f}s")
    print(f"  - External evaluation: {external_eval_elapsed:.1f}s")
    print(f"  - Total fold time: {fold_elapsed:.1f}s")
    
    threshold_results[-1]['total_fold_time'] = fold_elapsed
    threshold_results[-1]['per_sample_time'] = per_sample_elapsed
    threshold_results[-1]['external_eval_time'] = external_eval_elapsed

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
# Save external evaluation results (NEW)
if len(external_results) > 0:
    external_df = pd.DataFrame(external_results)
    external_df.to_csv(
        os.path.join(OUTPUT_PATH, "CV", "external_fcs_evaluation.csv"),
        index=False
    )
    print(f"\n[EXTERNAL RESULTS SAVED]")
    print(f"File: {os.path.join(OUTPUT_PATH, 'CV', 'external_fcs_evaluation.csv')}")
else:
    print(f"\n[WARNING] No external FCS files were successfully evaluated.")

# ============================
# ==========================================
# DIAGNOSTICS: Inner CV Grid Search Analysis for Fold 3  <-- ADD HERE
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("FOLD 3 DIAGNOSTICS: Inner CV Grid Search Analysis")
print("="*70)

# Load the inner CV results for fold 3
results_path = os.path.join(OUTPUT_PATH, "CV", "fold3_RandomForest_innerCV.csv")
cv_results = pd.read_csv(results_path)

# Print the best params and their validation F1
best_idx = cv_results['mean_test_score'].idxmax()
print("\nFold 3 CV Best Params:")
print(cv_results.loc[best_idx])

print(f"\nTop 5 hyperparameter configurations by F1:")
print(cv_results.nlargest(5, 'mean_test_score')[['param_max_depth', 'param_n_estimators', 
                                                   'param_min_samples_leaf', 'param_min_samples_split', 
                                                   'mean_test_score']])

# Plot 1: max_depth vs F1
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=cv_results,
    x='param_max_depth',
    y='mean_test_score',
    size='param_n_estimators',
    hue='param_min_samples_leaf',
    palette='viridis',
    s=100,
    alpha=0.7
)
plt.title("Fold 3: Random Forest Inner CV - max_depth vs F1 Score")
plt.ylabel("Mean CV F1 Score")
plt.xlabel("max_depth")
plt.grid(True, alpha=0.3)
plot1_path = os.path.join(OUTPUT_PATH, "CV", "fold3_grid_search_maxdepth_vs_f1.png")
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"\n? Saved: {plot1_path}")
plt.close()

# Plot 2: n_estimators vs F1
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=cv_results,
    x='param_n_estimators',
    y='mean_test_score',
    hue='param_max_depth',
    size='param_min_samples_leaf',
    palette='coolwarm',
    s=100,
    alpha=0.7
)
plt.title("Fold 3: Random Forest Inner CV - n_estimators vs F1 Score")
plt.ylabel("Mean CV F1 Score")
plt.xlabel("n_estimators")
plt.grid(True, alpha=0.3)
plot2_path = os.path.join(OUTPUT_PATH, "CV", "fold3_grid_search_nestimators_vs_f1.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"? Saved: {plot2_path}")
plt.close()

# Plot 3: Heatmap
pivot_table = cv_results.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdYlGn', cbar_kws={'label': 'Mean F1'})
plt.title("Fold 3: Inner CV F1 Heatmap - max_depth x n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plot3_path = os.path.join(OUTPUT_PATH, "CV", "fold3_grid_search_heatmap.png")
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"? Saved: {plot3_path}")
plt.close()

# Plot 4: F1 distribution
plt.figure(figsize=(10, 6))
plt.hist(cv_results['mean_test_score'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(cv_results['mean_test_score'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {cv_results["mean_test_score"].mean():.4f}')
plt.axvline(cv_results['mean_test_score'].max(), color='green', linestyle='--', 
            linewidth=2, label=f'Best: {cv_results["mean_test_score"].max():.4f}')
plt.title("Fold 3: Distribution of Inner CV F1 Scores")
plt.xlabel("Mean F1 Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plot4_path = os.path.join(OUTPUT_PATH, "CV", "fold3_f1_distribution.png")
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"? Saved: {plot4_path}")
plt.close()
# Save CSV summary
cv_results_summary = cv_results[['param_max_depth', 'param_n_estimators', 'param_min_samples_leaf',
                                  'param_min_samples_split', 'mean_test_score', 'std_test_score',
                                  'rank_test_score']].copy()
cv_results_summary = cv_results_summary.sort_values('mean_test_score', ascending=False)
cv_summary_path = os.path.join(OUTPUT_PATH, "CV", "fold3_grid_search_summary.csv")
cv_results_summary.to_csv(cv_summary_path, index=False)
print(f"? Saved: {cv_summary_path}")

print("\n[DIAGNOSTICS COMPLETE]")
print("="*70)

# ==========================================
# DIAGNOSTICS: Train/Test Composition Analysis for Fold 3
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("FOLD 3 DIAGNOSTICS: Train/Test Composition Analysis")
print("="*70)

# Set which fold to investigate
fold_of_interest = 3

# Get train/test indices for fold 3
outer_splits = list(outer_cv.split(X_scaled, y, groups))
train_idx, test_idx = outer_splits[fold_of_interest]

# Get DataFrames for each
train_df = data.iloc[train_idx].copy()
test_df = data.iloc[test_idx].copy()

# ==========================================
# DIAGNOSTICS 3: Threshold Curve Analysis
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import precision_recall_curve

print("\n" + "="*70)
print("THRESHOLD ANALYSIS: Precision/Recall/F1 Curves Across All Folds")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for fold in range(5):
    ax = axes[fold]
    
    # Load predictions for this fold
    pred_path = os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_predictions.pkl")
    
    try:
        with open(pred_path, 'rb') as f:
            data = pickle.load(f)
        
        y_test = data['y_true']
        proba_scores = data['proba_scores']
        optimal_threshold = data['optimal_threshold']
        
        # Calculate precision-recall-F1 curves
        precision, recall, thresholds = precision_recall_curve(y_test, proba_scores)
        f1s = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Plot curves
        ax.plot(thresholds, precision[:-1], label="Precision", linewidth=2.5, color='blue')
        ax.plot(thresholds, recall[:-1], label="Recall", linewidth=2.5, color='green')
        ax.plot(thresholds, f1s[:-1], label="F1", linewidth=2.5, color='purple')
        
        # Mark optimal threshold
        ax.axvline(optimal_threshold, color='red', linestyle='--', 
                  linewidth=2.5, label=f'Optimal: {optimal_threshold:.3f}')
        
        # Get corresponding metrics at optimal threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        
        ax.set_title(f"Fold {fold}: Threshold Analysis\nOptimal F1: {f1s[idx]:.3f}", 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        print(f"\n? Fold {fold}:")
        print(f"  - Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  - Precision @ optimal: {precision[idx]:.4f}")
        print(f"  - Recall @ optimal: {recall[idx]:.4f}")
        print(f"  - F1 @ optimal: {f1s[idx]:.4f}")
        
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"Fold {fold}\n\nPredictions not found.\nRe-run CV loop with\nprediction saving enabled.",
                ha='center', va='center', fontsize=11, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(f"Fold {fold}: (Data not available)", fontweight='bold')
        print(f"\n? Fold {fold}: Predictions not found (need to save during CV)")

# Remove extra subplot
fig.delaxes(axes[-1])

plt.suptitle("Threshold Analysis: Precision/Recall/F1 Curves Across All 5 Folds", 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_PATH, "CV", "threshold_analysis_all_folds.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n? Saved: {plot_path}")
plt.close()

# Optional: Generate individual fold plots with more detail
for fold in range(5):
    pred_path = os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_predictions.pkl")
    
    try:
        with open(pred_path, 'rb') as f:
            data = pickle.load(f)
        
        y_test = data['y_true']
        proba_scores = data['proba_scores']
        optimal_threshold = data['optimal_threshold']
        
        precision, recall, thresholds = precision_recall_curve(y_test, proba_scores)
        f1s = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Create individual plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2.5, marker='o', markersize=3)
        plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2.5, marker='s', markersize=3)
        plt.plot(thresholds, f1s[:-1], label="F1", linewidth=2.5, marker='^', markersize=3)
        
        plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2.5,
                   label=f'Optimal Threshold: {optimal_threshold:.3f}')
        
        plt.fill_between(thresholds, 0, 1, alpha=0.1, color='gray')
        
        plt.title(f"Fold {fold}: Detailed Threshold Analysis", fontweight='bold', fontsize=12)
        plt.xlabel("Decision Threshold", fontsize=11)
        plt.ylabel("Metric Value", fontsize=11)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        
        individual_path = os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_threshold_curve_detailed.png")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"? Saved: {individual_path}")
        plt.close()
        
    except FileNotFoundError:
        pass

print("\n[THRESHOLD ANALYSIS COMPLETE]")
print("="*70)

# ==========================================
# PART 1: Summary Statistics
# ==========================================
print("\n[SUMMARY STATISTICS]")
print(f"\nTrain Set:")
print(f"  - Total events: {len(train_df):,}")
print(f"  - Total blasts: {train_df['Blast'].sum():,} ({train_df['Blast'].mean():.2%})")
print(f"  - Total WBC: {(1-train_df['Blast']).sum():,} ({(1-train_df['Blast']).mean():.2%})")

print(f"\nTest Set:")
print(f"  - Total events: {len(test_df):,}")
print(f"  - Total blasts: {test_df['Blast'].sum():,} ({test_df['Blast'].mean():.2%})")
print(f"  - Total WBC: {(1-test_df['Blast']).sum():,} ({(1-test_df['Blast']).mean():.2%})")

print(f"\n[PATIENT DISTRIBUTION]")
train_patients = train_df['patient_id'].value_counts().sort_index()
test_patients = test_df['patient_id'].value_counts().sort_index()

print(f"\nTrain patients:")
for patient, count in train_patients.items():
    blast_count = train_df[train_df['patient_id']==patient]['Blast'].sum()
    print(f"  {patient}: {count:,} events ({blast_count:,} blasts, {blast_count/count:.2%})")

print(f"\nTest patients:")
for patient, count in test_patients.items():
    blast_count = test_df[test_df['patient_id']==patient]['Blast'].sum()
    print(f"  {patient}: {count:,} events ({blast_count:,} blasts, {blast_count/count:.2%})")

# ==========================================
# PART 2: Feature Distribution Comparison (Train vs Test)
# ==========================================
print(f"\n[FEATURE DISTRIBUTION STATISTICS]")
for feature in features:
    print(f"\n{feature}:")
    print(f"  Train WBC - Mean: {train_df[feature][train_df['Blast']==0].mean():.4f}, Std: {train_df[feature][train_df['Blast']==0].std():.4f}")
    print(f"  Train Blast - Mean: {train_df[feature][train_df['Blast']==1].mean():.4f}, Std: {train_df[feature][train_df['Blast']==1].std():.4f}")
    print(f"  Test WBC - Mean: {test_df[feature][test_df['Blast']==0].mean():.4f}, Std: {test_df[feature][test_df['Blast']==0].std():.4f}")
    print(f"  Test Blast - Mean: {test_df[feature][test_df['Blast']==1].mean():.4f}, Std: {test_df[feature][test_df['Blast']==1].std():.4f}")

# ==========================================
# PART 3: Visualize marker distributions per class
# ==========================================
print(f"\n[GENERATING DISTRIBUTION PLOTS]")

for feature in features:
    plt.figure(figsize=(10, 5))
    
    # KDE plots
    sns.kdeplot(
        data=train_df[train_df['Blast']==0][feature],
        label='Train WBC',
        fill=True,
        alpha=0.5,
        color='blue'
    )
    sns.kdeplot(
        data=train_df[train_df['Blast']==1][feature],
        label='Train Blast',
        fill=True,
        alpha=0.5,
        color='red'
    )
    sns.kdeplot(
        data=test_df[test_df['Blast']==0][feature],
        label='Test WBC',
        linestyle='--',
        linewidth=2,
        color='darkblue'
    )
    sns.kdeplot(
        data=test_df[test_df['Blast']==1][feature],
        label='Test Blast',
        linestyle='--',
        linewidth=2,
        color='darkred'
    )
    
    plt.title(f'Fold 3: {feature} - Train vs Test Distribution', fontsize=12, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(OUTPUT_PATH, "CV", f"fold3_composition_{feature}_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"? Saved: {plot_path}")
    plt.close()

# ==========================================
# PART 4: Boxplot comparison per feature
# ==========================================
print(f"\n[GENERATING BOXPLOT COMPARISON]")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(features):
    # Prepare data for boxplot
    data_for_plot = []
    labels_for_plot = []
    
    for label in ['Train WBC', 'Train Blast', 'Test WBC', 'Test Blast']:
        if 'Train' in label:
            subset = train_df[train_df['Blast'] == (1 if 'Blast' in label else 0)][feature]
        else:
            subset = test_df[test_df['Blast'] == (1 if 'Blast' in label else 0)][feature]
        
        data_for_plot.append(subset)
        labels_for_plot.append(label)
    
    # Create boxplot
    bp = axes[idx].boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightcoral', 'blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[idx].set_title(f'{feature}', fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True, alpha=0.3, axis='y')

# Remove extra subplot if odd number of features
if len(features) < len(axes):
    fig.delaxes(axes[-1])

plt.suptitle('Fold 3: Feature Distribution - Train vs Test (Boxplot)', fontsize=14, fontweight='bold')
plt.tight_layout()

boxplot_path = os.path.join(OUTPUT_PATH, "CV", "fold3_composition_boxplot_comparison.png")
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f"? Saved: {boxplot_path}")
plt.close()

# ==========================================
# PART 5: Patient balance visualization
# ==========================================
print(f"\n[GENERATING PATIENT BALANCE PLOTS]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Train patient distribution
train_patients.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
ax1.set_title('Fold 3: Train Set - Events per Patient', fontweight='bold')
ax1.set_xlabel('Patient ID')
ax1.set_ylabel('Number of Events')
ax1.grid(True, alpha=0.3, axis='y')

# Test patient distribution
test_patients.plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
ax2.set_title('Fold 3: Test Set - Events per Patient', fontweight='bold')
ax2.set_xlabel('Patient ID')
ax2.set_ylabel('Number of Events')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

patient_balance_path = os.path.join(OUTPUT_PATH, "CV", "fold3_composition_patient_balance.png")
plt.savefig(patient_balance_path, dpi=300, bbox_inches='tight')
print(f"? Saved: {patient_balance_path}")
plt.close()

# ==========================================
# PART 6: Blast prevalence per patient
# ==========================================
print(f"\n[GENERATING BLAST PREVALENCE BY PATIENT]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Train blast prevalence
train_blast_perc = []
train_patients_list = []
for patient in sorted(train_df['patient_id'].unique()):
    patient_data = train_df[train_df['patient_id']==patient]
    blast_perc = patient_data['Blast'].mean() * 100
    train_blast_perc.append(blast_perc)
    train_patients_list.append(patient)

ax1.bar(range(len(train_patients_list)), train_blast_perc, color='steelblue', alpha=0.7)
ax1.set_xticks(range(len(train_patients_list)))
ax1.set_xticklabels(train_patients_list, rotation=45)
ax1.set_title('Fold 3: Train Set - Blast % per Patient', fontweight='bold')
ax1.set_ylabel('Blast Prevalence (%)')
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=train_df['Blast'].mean()*100, color='red', linestyle='--', label='Overall Mean')
ax1.legend()

# Test blast prevalence
test_blast_perc = []
test_patients_list = []
for patient in sorted(test_df['patient_id'].unique()):
    patient_data = test_df[test_df['patient_id']==patient]
    blast_perc = patient_data['Blast'].mean() * 100
    test_blast_perc.append(blast_perc)
    test_patients_list.append(patient)

ax2.bar(range(len(test_patients_list)), test_blast_perc, color='coral', alpha=0.7)
ax2.set_xticks(range(len(test_patients_list)))
ax2.set_xticklabels(test_patients_list, rotation=45)
ax2.set_title('Fold 3: Test Set - Blast % per Patient', fontweight='bold')
ax2.set_ylabel('Blast Prevalence (%)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=test_df['Blast'].mean()*100, color='red', linestyle='--', label='Overall Mean')
ax2.legend()

plt.tight_layout()

blast_prev_path = os.path.join(OUTPUT_PATH, "CV", "fold3_composition_blast_prevalence.png")
plt.savefig(blast_prev_path, dpi=300, bbox_inches='tight')
print(f"? Saved: {blast_prev_path}")
plt.close()

# ==========================================
# PART 7: Save summary statistics to CSV
# ==========================================
print(f"\n[SAVING SUMMARY STATISTICS]")

composition_summary = {
    'Metric': [
        'Train Total Events', 'Train Total Blasts', 'Train Blast %',
        'Test Total Events', 'Test Total Blasts', 'Test Blast %',
        'Train/Test Ratio', 'Blast Prevalence Ratio (Train/Test)'
    ],
    'Value': [
        len(train_df),
        train_df['Blast'].sum(),
        f"{train_df['Blast'].mean():.2%}",
        len(test_df),
        test_df['Blast'].sum(),
        f"{test_df['Blast'].mean():.2%}",
        f"{len(train_df)/len(test_df):.2f}:1",
        f"{train_df['Blast'].mean()/test_df['Blast'].mean():.2f}"
    ]
}

composition_df = pd.DataFrame(composition_summary)
composition_csv_path = os.path.join(OUTPUT_PATH, "CV", "fold3_composition_summary.csv")
composition_df.to_csv(composition_csv_path, index=False)
print(f"? Saved: {composition_csv_path}")

print("\n[TRAIN/TEST COMPOSITION DIAGNOSTICS COMPLETE]")
print("="*70)

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

# Summary statistics for external evaluation (NEW)
if len(external_results) > 0:
    print(f"\n[EXTERNAL EVALUATION SUMMARY]")
    ext_df = pd.DataFrame(external_results)
    print(f"Total external files evaluated: {len(ext_df['external_file'].unique())}")
    print(f"Total evaluations: {len(ext_df)}")
    
    if 'precision' in ext_df.columns:
        labeled_ext = ext_df[ext_df['has_labels'] == True]
        if len(labeled_ext) > 0:
            print(f"Files with labels: {len(labeled_ext)}")
            print(f"  - Average Precision: {labeled_ext['precision'].mean():.3f} +/- {labeled_ext['precision'].std():.3f}")
            print(f"  - Average Recall: {labeled_ext['recall'].mean():.3f} +/- {labeled_ext['recall'].std():.3f}")
            print(f"  - Average F1: {labeled_ext['f1'].mean():.3f} +/- {labeled_ext['f1'].std():.3f}")
        
        unlabeled_ext = ext_df[ext_df['has_labels'] == False]
        if len(unlabeled_ext) > 0:
            print(f"Files without labels (predictions only): {len(unlabeled_ext)}")
            print(f"  - Average predicted blast %: {unlabeled_ext['pred_blast_perc'].mean():.2%}")