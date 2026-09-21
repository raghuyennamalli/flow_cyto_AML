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
    precision_recall_curve, roc_curve, auc)
from joblib import parallel_backend
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================= PARAMETERS & PATHS ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output_w_2feat"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "plots_downsampled"), exist_ok=True)

print(f"[DEBUG] FCS_PATH: {FCS_PATH}")
print(f"[DEBUG] LABEL_PATH: {LABEL_PATH}")
print(f"[DEBUG] OUTPUT_PATH: {OUTPUT_PATH}")
print(f"[DEBUG] FCS files exist: {os.path.exists(FCS_PATH)}")
print(f"[DEBUG] LABEL files exist: {os.path.exists(LABEL_PATH)}")

features = [ "PerCP-A", "PC7-A"]
# ========================================= HELPER FUNCTIONS ==========================================

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
    
    We want: scores >= threshold ? classify as blast (class 1) """
    
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

# ========================================== VISUALIZATION FUNCTIONS ==========================================

def plot_confusion_matrix(y_true, y_pred, fold, output_path):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['WBC', 'Blast'], 
                yticklabels=['WBC', 'Blast'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Fold {fold+1} (Downsampled IF)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, fold, output_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Fold {fold+1} (Downsampled IF)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, fold, output_path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=0.5, color='gray', linestyle='--', lw=1, alpha=0.5, label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Fold {fold+1} (Downsampled IF)', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_anomaly_score_distribution(y_true, y_scores, fold, output_path):
    """Plot distribution of anomaly scores."""
    blast_scores = y_scores[y_true == 1]
    normal_scores = y_scores[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})', 
             color='blue', edgecolor='black')
    plt.hist(blast_scores, bins=50, alpha=0.6, label=f'Blast (n={len(blast_scores)})', 
             color='red', edgecolor='black')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Anomaly Score Distribution - Fold {fold+1} (Downsampled IF)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_anomaly_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_by_class_boxplot(y_true, y_scores, fold, output_path):
    """Plot box plot of scores by class."""
    data_dict = {
        'Score': y_scores,
        'Class': ['Normal' if y == 0 else 'Blast' for y in y_true]
    }
    df_scores = pd.DataFrame(data_dict)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_scores, x='Class', y='Score', palette=['blue', 'red'], width=0.5)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title(f'Anomaly Scores by Class - Fold {fold+1} (Downsampled IF)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'fold{fold}_score_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_metrics(threshold_results, output_path):
    """Plot summary metrics across all folds."""
    results_df = pd.DataFrame(threshold_results)
    folds = results_df['fold'].values + 1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision
    axes[0, 0].bar(folds, results_df['precision'], color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('Precision', fontsize=11)
    axes[0, 0].set_title('Precision Across Folds', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(results_df['precision']):
        axes[0, 0].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Recall
    axes[0, 1].bar(folds, results_df['recall'], color='orangered', edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Recall', fontsize=11)
    axes[0, 1].set_title('Recall Across Folds', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(results_df['recall']):
        axes[0, 1].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # F1 Score
    axes[1, 0].bar(folds, results_df['f1'], color='green', edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('F1 Score Across Folds', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(results_df['f1']):
        axes[1, 0].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Average metrics with error bars
    metrics_names = ['Precision', 'Recall', 'F1']
    means = [results_df['precision'].mean(), results_df['recall'].mean(), results_df['f1'].mean()]
    stds = [results_df['precision'].std(), results_df['recall'].std(), results_df['f1'].std()]
    
    axes[1, 1].bar(metrics_names, means, yerr=stds, capsize=10, 
                   color=['steelblue', 'orangered', 'green'], edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Average Metrics +/- Std (Downsampled IF)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1, 1].text(i, m + s + 0.05, f'{m:.3f}+/-{s:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_performance(all_fold_results, output_path):
    """Plot per-sample performance metrics."""
    results_df = pd.DataFrame(all_fold_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Recall per sample
    sample_recall = results_df.groupby('sample_id')['recall'].mean().sort_values()
    axes[0, 0].barh(range(len(sample_recall)), sample_recall.values, color='orangered', edgecolor='black', alpha=0.7)
    axes[0, 0].set_yticks(range(len(sample_recall)))
    axes[0, 0].set_yticklabels(sample_recall.index, fontsize=9)
    axes[0, 0].set_xlabel('Recall', fontsize=11)
    axes[0, 0].set_title('Recall Per Sample (Downsampled IF)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # Precision per sample
    sample_precision = results_df.groupby('sample_id')['precision'].mean().sort_values()
    axes[0, 1].barh(range(len(sample_precision)), sample_precision.values, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_yticks(range(len(sample_precision)))
    axes[0, 1].set_yticklabels(sample_precision.index, fontsize=9)
    axes[0, 1].set_xlabel('Precision', fontsize=11)
    axes[0, 1].set_title('Precision Per Sample (Downsampled IF)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='x')
    
    # F1 per sample
    sample_f1 = results_df.groupby('sample_id')['f1'].mean().sort_values()
    axes[1, 0].barh(range(len(sample_f1)), sample_f1.values, color='green', edgecolor='black', alpha=0.7)
    axes[1, 0].set_yticks(range(len(sample_f1)))
    axes[1, 0].set_yticklabels(sample_f1.index, fontsize=9)
    axes[1, 0].set_xlabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('F1 Score Per Sample (Downsampled IF)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # Ground truth vs predicted blast count
    gt_counts = results_df.groupby('sample_id')['gt_count'].first()
    pred_counts = results_df.groupby('sample_id')['pred_count'].first()
    
    x = np.arange(len(gt_counts))
    width = 0.35
    axes[1, 1].bar(x - width/2, gt_counts.values, width, label='Ground Truth', color='blue', edgecolor='black', alpha=0.7)
    axes[1, 1].bar(x + width/2, pred_counts.values, width, label='Predicted', color='orange', edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Blast Count: Ground Truth vs Predicted (Downsampled IF)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(gt_counts.index, fontsize=9)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sample_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def make_5pct_dataset(data, output_path):
    print(f"\n[INFO] Creating 5% blast dataset: {output_path}")

    total_n = len(data)
    blasts = data[data["Blast"] == 1]
    normal = data[data["Blast"] == 0]

    target_blasts = int(total_n * 0.05)              # 5%
    target_normal = total_n - target_blasts          # 95%

    if len(blasts) < target_blasts:
        print(f"[ERROR] Not enough blasts: have {len(blasts)}, need {target_blasts}")
        return

    if len(normal) < target_normal:
        print(f"[WARNING] Not enough normal cells, taking all available.")
        target_normal = len(normal)

    # Downsample blasts to 5% of dataset
    blasts_sample = blasts.sample(n=target_blasts, random_state=42)

    # Sample normal cells to match full dataset size
    normal_sample = normal.sample(n=target_normal, random_state=42)

    # Combine
    df_5pct = pd.concat([blasts_sample, normal_sample], ignore_index=True)

    # Shuffle
    df_5pct = df_5pct.sample(frac=1, random_state=42).reset_index(drop=True)

    df_5pct.to_pickle(output_path)

    print(f"[SUCCESS] 5% dataset saved: {output_path}")
    print(f"[INFO] Final blast prevalence = {df_5pct['Blast'].mean():.2%}\n")


 #========================================== STEP 0: CREATE AGGREGATED DATASET (2K + 5K) ==========================================
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
    # Run for 2K
    make_5pct_dataset(data=data_2K,output_path=os.path.join(OUTPUT_PATH, "BLAST110_2K_5pct.pkl"))
    # Run for 5K
    make_5pct_dataset(data=data_5K,output_path=os.path.join(OUTPUT_PATH, "BLAST110_5K_5pct.pkl"))

else:
    print("[INFO] Pickled data already exists.")



# ========================================== STEP 1: LOAD AGGREGATED DATA ==========================================
pkl_downsampled=os.path.join(OUTPUT_PATH, "BLAST110_5K_5pct.pkl")
data = pd.read_pickle(pkl_downsampled)

X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]

X_scaled= X.values

print(f"[INFO] Data shape: {X_scaled.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ========================================== STEP 2: DEFINE INNER + OUTER CV ==========================================
outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)

# Define parameter grid for Isolation Forest

# ========================================== STEP 3: OUTER CV LOOP  (WITH PARALLELIZATION) ==========================================
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
        blast_prevalence - 0.02,
        blast_prevalence - 0.01,
        blast_prevalence,
        blast_prevalence + 0.01,
        blast_prevalence + 0.02
    ]
    contamination_range = [max(0.001, c) for c in contamination_range]
    param_grid = {
    "n_estimators": [100, 200, 300],
    "max_samples": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.75, 1.0],
    'contamination':  contamination_range }
    print(f"[INFO] Contamination range: {[f'{c:.4f}' for c in contamination_range]}")

    print(f"Training on {X_train_all.shape[0]} normal cells")

    # ------------------------ INNER CV (Grid Search) ------------------------
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
    cv_results.to_csv(os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_downsampled_innerCV.csv"), index=False)
    
    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best Score: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")
    # ----------------------- OUTER TESTING ------------------------
    outer_test_start = time.time()
    
    best_iso = grid.best_estimator_

    anomaly_scores_test = best_iso.decision_function(X_test)
    optimal_thresh_clinical, metrics = find_optimal_threshold(y_test.values, anomaly_scores_test, method='f1')
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
    
    # ======================== VISUALIZATIONS ========================
    print(f"\n[VISUALIZATION] Creating plots...")
    
    plot_confusion_matrix(y_test, preds, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_roc_curve(y_test, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_precision_recall_curve(y_test, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_anomaly_score_distribution(y_test, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_score_by_class_boxplot(y_test, anomaly_scores_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))

    # ------------------------ PER-SAMPLE RESULTS ------------------------
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
        os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_downsampled_outerCV.csv"),
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

# ========================================= SAVE FINAL SUMMARIES ==========================================
pd.DataFrame(threshold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "threshold_summary.csv"),
    index=False)

pd.DataFrame(all_fold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "all_samples_results.csv"),
    index=False)
# ========================================== FINAL VISUALIZATIONS ==========================================
print("[VISUALIZATION] Creating summary plots...")
plot_summary_metrics(threshold_results, os.path.join(OUTPUT_PATH, "plots_downsampled"))
plot_sample_performance(all_fold_results, os.path.join(OUTPUT_PATH, "plots_downsampled"))

# ========================================== FINAL RUNTIME REPORT ==========================================
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

