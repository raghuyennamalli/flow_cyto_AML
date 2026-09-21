import os
import time
import numpy as np
import pandas as pd
import re  
from FlowCytometryTools import FCMeasurement
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer,
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, auc)
from joblib import parallel_backend
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================= PARAMETERS & PATHS ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output_IF6/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV_IF"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "plots_downsampled_IF"), exist_ok=True)

print(f"[DEBUG] FCS_PATH: {FCS_PATH}")
print(f"[DEBUG] LABEL_PATH: {LABEL_PATH}")
print(f"[DEBUG] OUTPUT_PATH: {OUTPUT_PATH}")
print(f"[DEBUG] FCS files exist: {os.path.exists(FCS_PATH)}")
print(f"[DEBUG] LABEL files exist: {os.path.exists(LABEL_PATH)}")

features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]
# ========================================= HELPER FUNCTIONS ==========================================

def from_fcs(path):
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()

def iforest_auprc_continuous(estimator, X, y):
    """
    Score-based AUPRC for IF hyperparameter tuning (continuous scores).
    
    Isolation Forest convention:
    - decision_function: higher = more normal (WBC), lower = anomalous (Blast)
    - For AUPRC: we need y_true=1 → high score
    - Negation converts: -score means blasts (y=1) get higher negated scores
    """
    y_scores_if = estimator.decision_function(X)
    y_scores_negated = -y_scores_if  # Negate so blasts get high scores for AUPRC
    return average_precision_score(y, y_scores_negated)

def find_optimal_threshold(y_true, y_scores, method='clinical_recall'):
    """
    Find optimal threshold for Isolation Forest decision function.
    
    Isolation Forest convention:
    - Higher scores = more normal (WBC)
    - Lower scores = more anomalous (Blast)
    - We want: scores <= threshold → classify as blast (class 1)
    
    Methods:
    - 'clinical_recall': Maximize recall while ensuring precision >= 0.90
    - 'f1': Maximize F1 score (balanced precision/recall)
    """
    y_scores_inverted = -y_scores
    
    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
        thresholds = np.append(thresholds, 0)
        fscore = (2 * precision * recall) / (precision + recall + 1e-10)
        ix = np.argmax(fscore)
        optimal_thresh = -thresholds[ix]
        
    elif method == 'clinical_recall':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
        thresholds = np.append(thresholds, 0)
        valid_idx = precision >= 0.90
        if np.sum(valid_idx) > 0:
            valid_recall = recall[valid_idx]
            valid_thresholds = thresholds[valid_idx]
            ix = np.argmax(valid_recall)
            optimal_thresh = -valid_thresholds[ix]
        else:
            fscore = (2 * precision * recall) / (precision + recall + 1e-10)
            ix = np.argmax(fscore)
            optimal_thresh = -thresholds[ix]
    else:  
        optimal_thresh = np.median(y_scores)
    
    y_pred = (y_scores <= optimal_thresh).astype(int)
    
    return optimal_thresh, {
        'threshold': optimal_thresh,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

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

# ========================================== VISUALIZATION FUNCTIONS ==========================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, fold, output_path):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Blast', 'Blast'], 
                yticklabels=['Non-Blast', 'Blast'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12)
    save_fig(os.path.join(output_path, f'fold{fold}_confusion_matrix.png'))

def plot_roc_curve(y_true, y_scores, fold, output_path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Predicts Blasts but actually non-blasts)', fontsize=12)
    plt.ylabel('True Positive Rate (Correctly predict Blast)', fontsize=12)
    plt.title(f'ROC Curve - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    save_fig(os.path.join(output_path, f'fold{fold}_roc_curve.png'))
    

def plot_precision_recall_curve(y_true, y_scores, fold, output_path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=0.5, color='gray', linestyle='--', lw=1, alpha=0.5, label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11); plt.grid(alpha=0.3)
    save_fig(os.path.join(output_path, f'fold{fold}_precision_recall_curve.png'))

def plot_anomaly_score_distribution(y_true, y_scores, fold, output_path):
    """Plot distribution of anomaly scores."""
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label=f'Non-blasts (n={len(y_scores[y_true == 0])})', color='blue', edgecolor='black')
    plt.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label=f'Blast (n={len(y_scores[y_true == 1])})', color='red', edgecolor='black')
    plt.xlabel('Anomaly Score', fontsize=12); plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Anomaly Score Distribution - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11); plt.grid(alpha=0.3, axis='y')
    save_fig(os.path.join(output_path, f'fold{fold}_anomaly_score_distribution.png'))

def plot_score_by_class_boxplot(y_true, y_scores, fold, output_path):
    """Plot box plot of scores by class."""
    df_scores = pd.DataFrame({'Score': y_scores,
        'Class': ['Non-blasts' if y == 0 else 'Blast' for y in y_true]})
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_scores, x='Class', y='Score', palette=['blue', 'red'], width=0.5)
    plt.xlabel('Class', fontsize=12); plt.ylabel('Anomaly Score', fontsize=12)
    plt.title(f'Anomaly Scores by Class - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    save_fig(os.path.join(output_path, f'fold{fold}_score_boxplot.png'))

def plot_summary_metrics(threshold_results, output_path):
    df = pd.DataFrame(threshold_results)
    folds = df['fold'] + 1
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['outer_precision', 'outer_recall', 'outer_f1']
    colors = ['steelblue', 'orangered', 'green']
    
    for ax, m, c in zip(axes.flat[:3], metrics, colors):
        metric_name = m.replace('outer_', '').capitalize()  # ← Moved inside loop
        ax.bar(folds, df[m], color=c, edgecolor='black', alpha=0.7)
        ax.set_title(f'{m.capitalize()} Across Folds', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate(df[m]):
            ax.text(folds.iloc[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    # Mean ± Std
    means = df[metrics].mean()
    stds = df[metrics].std()

    axes[1, 1].bar(metrics, means, yerr=stds, capsize=8,
                   color=colors, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Average Metrics ± Std', fontweight='bold')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(axis='y', alpha=0.3)

    save_fig(os.path.join(output_path, 'summary_metrics.png'))

def plot_sample_performance(all_results, output_path):
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Recall, Precision, F1 loop
    metrics = ['recall', 'precision', 'f1']
    colors = ['orangered', 'steelblue', 'green']

    for ax, m, c in zip(axes.flat[:3], metrics, colors):
        vals = df.groupby('sample_id')[m].mean().sort_values()
        ax.barh(vals.index, vals.values, color=c, edgecolor='black', alpha=0.7)
        ax.set_title(f'{m.capitalize()} Per Sample', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    # 4) Ground truth vs predicted
    gt = df.groupby('sample_id')['gt_count'].first()
    pred = df.groupby('sample_id')['pred_count'].first()

    idx = np.arange(len(gt))
    width = 0.35
    ax = axes[1, 1]
    ax.bar(idx - width/2, gt, width, label='Ground Truth')
    ax.bar(idx + width/2, pred, width, label='Predicted')

    ax.set_title('Blast Count: Ground Truth vs Predicted', fontweight='bold')
    ax.set_xticks(idx); ax.set_xticklabels(gt.index, fontsize=8)
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    save_fig(os.path.join(output_path, 'sample_performance.png'))

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
            name = re.sub(r"_Cleaned.*|_cleaned.*|_Scaled.*|_scaled.*|_transformed.*", "", name)

            parts = name.split("_")
            sample_id = parts[-1]
            patient_id = "_".join(parts[:-1])

            fcs_path = os.path.join(root, file)
            label_file = os.path.join(LABEL_PATH, f"{patient_id}_{sample_id}.csv")

            if not os.path.exists(label_file):
                print(f"{label_file} - Label file missing, skipping.")
                continue

            ff = from_fcs(fcs_path)
            labels = pd.read_csv(label_file, index_col=0)
            ff = pd.merge(ff, labels, on="event_ID")

            ff = ff[features + ["Blast", "Singlets", "event_ID"]]

            ff = ff[(ff["Singlets"] == 1)]
            ff = ff.drop(columns=["Singlets"])

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
inner_cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
outer_cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)

# ========================================== STEP 3: OUTER CV LOOP  (WITH PARALLELIZATION) ==========================================
total_start_time = time.time()
threshold_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y, groups=groups)):
    print(f"\n{'='*30} OUTER FOLD {fold+1}/{outer_cv.get_n_splits()} {'='*30}")
    
    fold_start_time = time.time()

    X_train = X_scaled[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_scaled[test_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    blast_prevalence = y_train.mean()
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")
    print(f"Train blast %: {y_train.mean():.2%}, Test blast %: {y_test.mean():.2%}")
    
    contamination_range = [
    blast_prevalence - 0.03,    # Lower bound: 1%
    0.01, blast_prevalence - 0.015,
    blast_prevalence,
    min(0.5, blast_prevalence + 0.03) ]
    contamination_range = [max(0.001, c) for c in contamination_range]
    param_grid = {
    "n_estimators": [100, 200],
    "max_samples": [0.3,0.5,  1.0],
    "max_features": [0.44, 0.5, 0.75,1.0],
    'contamination':  contamination_range }
    print(f"[INFO] Contamination range: {[f'{c:.4f}' for c in contamination_range]}")

    # ------------------------ INNER CV (Grid Search) ------------------------
    print(f"\n[INNER CV] Starting grid search with parallelization...")
    inner_cv_start = time.time() # Adjust based on available cores
    base_iso = IsolationForest( random_state=fold, n_jobs=1) # Parallelize tree building
    grid = GridSearchCV(
            base_iso,
            param_grid,
            cv=inner_cv,
            scoring= iforest_auprc_continuous,
            n_jobs=-1,  # Parallel hyperparameter search (safe with threading)
            refit=True,
            verbose=0)
    
    start = time.time()
    grid.fit(X_train, y_train, groups=train_groups)  
    elapsed_inner = time.time() - start
    inner_cv_elapsed = time.time() - inner_cv_start

    # Save inner CV results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_innerCV.csv"), index=False)
   
    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best AUPRC: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")
    
    
     # ======================== THRESHOLD DETERMINATION (PER-FOLD) ========================
    print(f"\n[THRESHOLD DETERMINATION] Computing fold-specific threshold...")
    inner_scores_dev = []
    inner_labels_dev = []
    n_valid_folds = 0
    
    # Re-score all inner validation folds to collect development data
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups=train_groups):
        if len(inner_val_idx) == 0:  
           print("[WARNING] Inner fold empty, skipping")
           continue
           
        y_val = y_train.iloc[inner_val_idx].values
        if np.sum(y_val) == 0:  # Add this guard
            print(f"[WARNING] Inner fold {fold} has no blasts, skipping")
            continue
            
        X_val = X_train[inner_val_idx]
        if X_val.shape[0] == 0:
            continue
            
        iso_temp = IsolationForest(**grid.best_params_, random_state=fold, n_jobs=-1)
        iso_temp.fit(X_train[inner_train_idx])
        
        scores_inner = iso_temp.decision_function(X_val)
        inner_scores_dev.extend(scores_inner)
        inner_labels_dev.extend(y_val)
        n_valid_folds += 1
        
        print(f"  [COLLECT] Inner fold {n_valid_folds}: {len(X_val):,} samples, "
          f"{np.sum(y_val):,} blasts, cumulative: {len(inner_scores_dev):,} samples")
    
    print(f"\n[THRESHOLD DETERMINATION] Computing fold-specific threshold...")
    print(f"  Total inner samples collected: {len(inner_scores_dev)}")
    print(f"  Valid inner folds: {n_valid_folds}/5")
    if len(inner_scores_dev) > 0:
        print(f"[INNER FOLD STATS] Blast prevalence: {np.mean(inner_labels_dev):.3%}")
        print(f"[INNER FOLD STATS] Score range: [{np.min(inner_scores_dev):.4f}, {np.max(inner_scores_dev):.4f}]")
    
    threshold_metrics = {
    'threshold': None,
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0,
    'method': 'unknown'}
        #Safeguard against empty development set:
    n_dev=len(inner_scores_dev)    
    
    if n_dev==0:
        print(f"\n[WARNING] No inner validation samples collected for fold {fold}.")
        print("[WARNING] Falling back to neutral threshold tau_fold = 0.0")
        tau_fold = 0.0
        threshold_metrics['method'] = 'no_data_fallback'
    
    elif len(inner_scores_dev) < 50:
           print(f"\n[WARNING] Insufficient inner validation samples ({len(inner_scores_dev)})")
           print(f"[WARNING] Using fallback strategy: test set median")
           
           tau_fold = np.median(inner_scores_dev)
           
           y_pred_inner = (np.array(inner_scores_dev) <= tau_fold).astype(int)
           threshold_metrics = {
            'threshold': tau_fold,
            'precision': precision_score(inner_labels_dev, y_pred_inner, zero_division=0),
            'recall': recall_score(inner_labels_dev, y_pred_inner, zero_division=0),
            'f1': f1_score(inner_labels_dev, y_pred_inner, zero_division=0), 
            'method': 'inner_median_fallback'}
           
           print(f"[THRESHOLD] Fold {fold+1} fallback threshold (test median): {tau_fold:.4f}")
           print(f"  Precision at threshold: {threshold_metrics['precision']:.3f}")
           print(f"  Recall at threshold: {threshold_metrics['recall']:.3f}")
           print(f"  F1 at threshold: {threshold_metrics['f1']:.3f}")
           
    else:
           print(f"\n[NORMAL] Sufficient inner validation data ({len(inner_scores_dev)} samples from {n_valid_folds} folds)")
           print(f"[NORMAL] Using F1-optimized threshold from inner CV data")
           tau_fold, threshold_metrics = find_optimal_threshold(np.array(inner_labels_dev),np.array(inner_scores_dev),method='f1')
          
    print(f"[THRESHOLD] Fold {fold+1} optimal threshold (inner CV): {tau_fold:.4f}")
    print(f"  Precision at threshold: {threshold_metrics['precision']:.3f}")
    print(f"  Recall at threshold: {threshold_metrics['recall']:.3f}")
    print(f"  F1 at threshold: {threshold_metrics['f1']:.3f}")
    
    # ----------------------- OUTER TESTING ------------------------
    outer_test_start = time.time()
    
    best_iso = grid.best_estimator_

    y_scores_test_if = best_iso.decision_function(X_test)
    y_scores_test_negated = -y_scores_test_if
    # PRIMARY: Score-based metrics (threshold-independent)
    outer_auprc = average_precision_score(y_test, y_scores_test_negated)
    outer_auroc = roc_auc_score(y_test, y_scores_test_negated)
    
    print(f"\n[OUTER CV] Score-Based Metrics (PRIMARY):")
    print(f"  AUPRC: {outer_auprc:.4f}")
    print(f"  AUROC: {outer_auroc:.4f}")
    
    # ======================== OUTER TEST (Operational metrics) =======================
    y_pred_test = (y_scores_test_if <= tau_fold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    outer_test_elapsed = time.time() - outer_test_start
    
    # ✓ FIX #8: Print fold-specific threshold
    print(f"\n[OUTER CV] SECONDARY - Operational Metrics (at threshold={tau_fold:.4f}):")
    print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]:<6} FP={cm[0,1]:<6}]")
    print(f"     [FN={cm[1,0]:<6} TP={cm[1,1]:<6}]]")


    # ======================== VISUALIZATIONS ========================
    print(f"\n[VISUALIZATION] Creating plots...")

    
    plot_confusion_matrix(y_test, y_pred_test, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_roc_curve(y_test, y_scores_test_negated, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_precision_recall_curve(y_test, y_scores_test_negated, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_anomaly_score_distribution(y_test, y_scores_test_if, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))
    plot_score_by_class_boxplot(y_test, y_scores_test_if, fold, os.path.join(OUTPUT_PATH, "plots_downsampled"))

    fold_elapsed= time.time() - fold_start_time
    threshold_results.append({
        'fold': fold,
        'best_params': str(grid.best_params_),
        'blast_prevalence_train': blast_prevalence,
        'n_inner_folds_valid': n_valid_folds,
        'n_inner_samples_collected': len(inner_scores_dev),
        'threshold_method': threshold_metrics.get('method', 'unknown'),
        'outer_auprc': outer_auprc,
        'outer_auroc': outer_auroc,
        'threshold_fold_specific': tau_fold,
        'threshold_precision': threshold_metrics['precision'],
        'threshold_recall': threshold_metrics['recall'],
        'threshold_f1': threshold_metrics['f1'],
        'outer_precision': prec,
        'outer_recall': rec,
        'outer_f1': f1,
        'inner_cv_time': inner_cv_elapsed,
        'outer_test_time': outer_test_elapsed,
        'total_fold_time': fold_elapsed})
    
    print(f"\n[FOLD TIMING]")
    print(f"  - Inner CV: {inner_cv_elapsed:.1f}s")
    print(f"  - Outer test: {outer_test_elapsed:.1f}s")
    print(f"  - Total fold time: {fold_elapsed:.1f}s")

print("\n" + "="*50 + " Nested Cross-Validation Complete " + "="*50)
# Report threshold stability
thresholds = [r['threshold_fold_specific'] for r in threshold_results]
print(f"\n[THRESHOLD STABILITY]")
print(f"Mean ± std: {np.mean(thresholds):.4f} +/-{np.std(thresholds):.4f}")
print(f"Range: [{np.min(thresholds):.4f}, {np.max(thresholds):.4f}]")

# ========================================= SAVE FINAL SUMMARIES ==========================================
pd.DataFrame(threshold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "threshold_summary.csv"),
    index=False)
# ========================================== FINAL VISUALIZATIONS ==========================================
print("[VISUALIZATION] Creating summary plots...")
plot_summary_metrics(threshold_results, os.path.join(OUTPUT_PATH, "plots_downsampled"))

# ========================================== FINAL RUNTIME REPORT ==========================================
total_elapsed = time.time() - total_start_time

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

print(f"\n[PERFORMANCE SUMMARY]")
print(f"Average Precision: {avg_results['outer_precision'].mean():.3f} ± {avg_results['outer_precision'].std():.3f}")
print(f"Average Recall: {avg_results['outer_recall'].mean():.3f} ± {avg_results['outer_recall'].std():.3f}")
print(f"Average F1: {avg_results['outer_f1'].mean():.3f} ± {avg_results['outer_f1'].std():.3f}")
print(f"Average AUPRC: {avg_results['outer_auprc'].mean():.3f} ± {avg_results['outer_auprc'].std():.3f}")
print(f"Average AUROC: {avg_results['outer_auroc'].mean():.3f} ± {avg_results['outer_auroc'].std():.3f}")

print(f"\n[THRESHOLD VARIATION]")
print(f"Mean fold-specific threshold: {avg_results['threshold_fold_specific'].mean():.4f}")
print(f"Threshold std dev: {avg_results['threshold_fold_specific'].std():.4f}")
print(f"Threshold range: [{avg_results['threshold_fold_specific'].min():.4f}, {avg_results['threshold_fold_specific'].max():.4f}]")

print(f"\n[OUTPUT FILES]")
print(f"Threshold summary: {os.path.join(OUTPUT_PATH, 'CV', 'threshold_summary.csv')}")