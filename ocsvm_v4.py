import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import  KFold, GridSearchCV, ParameterGrid, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer,
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, auc)
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
import pickle
# ========================================= SETUP ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output_svm"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV_v4_full"), exist_ok=True)
CV_PATH = os.path.join(OUTPUT_PATH, "CV_v4_full")

features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]

start_time = time.time()
script_start_time = time.time()
# ========================================== HELPER FUNCTIONS ==========================================
def ocsvm_scorer(estimator, X, y):
        scores = estimator.decision_function(X)
        if len(np.unique(y)) > 1:
            return roc_auc_score(y, -scores)
        return 0.5

def from_fcs(path):
    """Load FCS file and return data as a pandas DataFrame."""
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()
    
def optimal_threshold_f1(y_true, y_scores_inverted):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
    thresholds = np.append(thresholds, 0)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    return -thresholds[ix], precision, recall

def find_optimal_threshold(y_true, y_scores, method='clinical_recall'):
    y_scores_inverted = -y_scores
    
    if method == 'f1':
        optimal_thresh, precision, recall = optimal_threshold_f1(y_true, y_scores_inverted)
        
    elif method == 'clinical_recall':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
        thresholds = np.append(thresholds, 0)
        valid_idx = precision >= 0.90
        if np.sum(valid_idx) > 0:
            valid_thresholds = thresholds[valid_idx]
            ix = np.argmax(recall[valid_idx])
            optimal_thresh = -valid_thresholds[ix]
        else:
            optimal_thresh, precision, recall = optimal_threshold_f1(y_true, y_scores_inverted)
    else:  
        print("NO METHOD SPECIFIED SO WENT WITH F1")
        optimal_thresh, precision, recall = optimal_threshold_f1(y_true, y_scores_inverted)
    y_pred = (y_scores <= optimal_thresh).astype(int)
    
    return optimal_thresh, {'threshold': optimal_thresh,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)}
        
# ========================================== VISUALIZATION FUNCTIONS ==========================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, fold, output_path):     
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Blast', 'Blast'], yticklabels=['Non-Blast', 'Blast'], cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics_text = f'TP={tp}, TN={tn}, FP={fp}, FN={fn}\nSensitivity={sensitivity:.3f}, Specificity={specificity:.3f}'
    plt.text(1.0, -0.6, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    save_fig(os.path.join(output_path, f'fold{fold}_confusion_matrix.png'))
    
def plot_roc_curve(y_true, y_scores_negated, fold, output_path): #ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores_negated)
    roc_auc = auc(fpr, tpr)
        
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Predicts Blasts but actually non-blasts)', fontsize=12) ; plt.ylabel('True Positive Rate (Correctly predict Blast)', fontsize=12)
    plt.title(f'ROC Curve - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    save_fig(os.path.join(output_path, f'fold{fold}_roc_curve.png'))
    
def plot_precision_recall_curve(y_true, y_scores_negated, fold, output_path):     #Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores_negated)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=0.5, color='gray', linestyle='--', lw=1, alpha=0.5, label='No Skill')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11); plt.grid(alpha=0.3)
    save_fig(os.path.join(output_path, f'fold{fold}_precision_recall_curve.png'))

def plot_decision_score_distribution(y_true, y_scores_negated, fold, output_path):
    """
    Plot distribution of negated decision function scores by class
    Shows separation between normal and anomalous samples
    """
    nonblast_scores = y_scores_negated[y_true == 0]
    blast_scores = y_scores_negated[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(nonblast_scores, bins=40, alpha=0.6, label=f'Non-Blasts (n={len(nonblast_scores)})', color='#4ECDC4', edgecolor='black')
    plt.hist(blast_scores, bins=40, alpha=0.6, label=f'Blasts (n={len(blast_scores)})', color='#FF6B6B', edgecolor='black')
    
    plt.xlabel('Decision Function Score (Negated)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Decision Score Distribution - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = (f"Non-Blast: μ={np.mean(nonblast_scores):.3f}, σ={np.std(nonblast_scores):.3f}\n"
                  f"Blast: μ={np.mean(blast_scores):.3f}, σ={np.std(blast_scores):.3f}")
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    save_fig(os.path.join(output_path, f'fold{fold}_decision_score_distribution.png'))

def plot_score_by_class_boxplot(y_true, y_scores_negated, fold, output_path):
    """
    Plot box plot of decision scores by class
    Shows median, quartiles, and outliers
    """
    df_scores = pd.DataFrame({'Score': y_scores_negated, 'Class': ['Non-Blast' if y == 0 else 'Blast' for y in y_true]})
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_scores, x='Class', y='Score', palette=['#4ECDC4', '#FF6B6B'], width=0.5,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='darkred', linewidth=2.5))
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Decision Function Score (Negated)', fontsize=12, fontweight='bold')
    plt.title(f'Decision Scores by Class (Boxplot) - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    
    save_fig(os.path.join(output_path, f'fold{fold}_score_boxplot.png'))
    
def plot_threshold_line_on_distribution(y_true, y_scores_negated, threshold, fold, output_path):
    """Plot decision score distribution with threshold line overlay"""
    nonblast_scores = y_scores_negated[y_true == 0]
    blast_scores = y_scores_negated[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(nonblast_scores, bins=40, alpha=0.6, label=f'Non-Blasts (n={len(nonblast_scores)})', 
             color='#4ECDC4', edgecolor='black')
    plt.hist(blast_scores, bins=40, alpha=0.6, label=f'Blasts (n={len(blast_scores)})', 
             color='#FF6B6B', edgecolor='black')
    
    plt.axvline(-threshold, color='red', linestyle='--', linewidth=3, 
                label=f'Threshold (τ={threshold:.4f})', alpha=0.8)
    
    plt.xlabel('Decision Function Score (Negated)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Decision Scores with Threshold - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, axis='y')
    
    save_fig(os.path.join(output_path, f'fold{fold}_distribution_with_threshold.png'))
    
def plot_summary_metrics(per_fold_visualizations, output_path):
    #Plot summary metrics across all folds
    
    folds = [d['fold'] + 1 for d in per_fold_visualizations]
    auprcs = [d['auprc'] for d in per_fold_visualizations]
    aurocs = [d['auroc'] for d in per_fold_visualizations]
    precisions = [d['precision'] for d in per_fold_visualizations]
    recalls = [d['recall'] for d in per_fold_visualizations]
    f1s = [d['f1'] for d in per_fold_visualizations]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: AUPRC
    axes[0, 0].bar(folds, auprcs, color='#FF6B6B', edgecolor='black', alpha=0.7)
    axes[0, 0].axhline(np.mean(auprcs), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(auprcs):.3f}')
    axes[0, 0].set_title('AUPRC Across Folds', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(alpha=0.3, axis='y')
    axes[0, 0].legend()
    for i, v in enumerate(auprcs):
        axes[0, 0].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot 2: AUROC
    axes[0, 1].bar(folds, aurocs, color='#4ECDC4', edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(np.mean(aurocs), color='darkcyan', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(aurocs):.3f}')
    axes[0, 1].set_title('AUROC Across Folds', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(alpha=0.3, axis='y')
    axes[0, 1].legend()
    for i, v in enumerate(aurocs):
        axes[0, 1].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot 3: F1
    axes[0, 2].bar(folds, f1s, color='#95E1D3', edgecolor='black', alpha=0.7)
    axes[0, 2].axhline(np.mean(f1s), color='darkgreen', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(f1s):.3f}')
    axes[0, 2].set_title('F1 Score Across Folds', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(alpha=0.3, axis='y')
    axes[0, 2].legend()
    for i, v in enumerate(f1s):
        axes[0, 2].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot 4: Precision
    axes[1, 0].bar(folds, precisions, color='#FFB6C1', edgecolor='black', alpha=0.7)
    axes[1, 0].axhline(np.mean(precisions), color='darkred', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(precisions):.3f}')
    axes[1, 0].set_title('Precision Across Folds', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(alpha=0.3, axis='y')
    axes[1, 0].legend()
    for i, v in enumerate(precisions):
        axes[1, 0].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot 5: Recall
    axes[1, 1].bar(folds, recalls, color='#98D8C8', edgecolor='black', alpha=0.7)
    axes[1, 1].axhline(np.mean(recalls), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(recalls):.3f}')
    axes[1, 1].set_title('Recall Across Folds', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(alpha=0.3, axis='y')
    axes[1, 1].legend()
    for i, v in enumerate(recalls):
        axes[1, 1].text(folds[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot 6: Summary table
    axes[1, 2].axis('off')
    summary_data = [['Metric', 'Mean', 'Std', 'Min', 'Max'],
        ['AUPRC', f'{np.mean(auprcs):.3f}', f'{np.std(auprcs):.3f}', 
         f'{np.min(auprcs):.3f}', f'{np.max(auprcs):.3f}'],
        ['AUROC', f'{np.mean(aurocs):.3f}', f'{np.std(aurocs):.3f}', 
         f'{np.min(aurocs):.3f}', f'{np.max(aurocs):.3f}'],
        ['F1', f'{np.mean(f1s):.3f}', f'{np.std(f1s):.3f}', 
         f'{np.min(f1s):.3f}', f'{np.max(f1s):.3f}'],
        ['Precision', f'{np.mean(precisions):.3f}', f'{np.std(precisions):.3f}', 
         f'{np.min(precisions):.3f}', f'{np.max(precisions):.3f}'],
        ['Recall', f'{np.mean(recalls):.3f}', f'{np.std(recalls):.3f}', 
         f'{np.min(recalls):.3f}', f'{np.max(recalls):.3f}'] ]
    
    table = axes[1, 2].table(cellText=summary_data, cellLoc='center', loc='center',colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(5):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    fig.suptitle('Summary Metrics Across All Folds', fontsize=16, fontweight='bold')
    save_fig(os.path.join(output_path, 'summary_metrics.png'))


# ========================================== STEP 0: CREATE AGGREGATED DATASET (2K + 5K) ==========================================
pkl_2k = os.path.join(OUTPUT_PATH, "BLAST110_2K_5pct.pkl")
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
else:
    print("[INFO] Pickled data already exists.")
    
# ========================================== STEP 1: LOAD DATA (CORRECTED) ==========================================
pkl_2k = os.path.join(OUTPUT_PATH,"BLAST110_2K_5pct.pkl")
data = pd.read_pickle(pkl_2k)

X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]

X_all = X.values

print(f"[INFO] Data shape: {X_all.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ========================================== # STEP 2: DEFINE CV STRATEGY # ==========================================
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


# ========================================== STEP 3: EXHAUSTIVE HYPERPARAMETER GRID ==========================================

total_start = time.time()
all_fold_threshold = []
all_outer_fold_results=[]
per_fold_visualizations = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y, groups=groups)):

    print(f"\n{'='*30}OUTER FOLD {fold + 1}{'='*30}")
    fold_start = time.time()
    # ---- Split data for this outer fold ----
    X_train = X_all[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X_all[test_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    blast_rate_test = y_test.mean()
    

    #X_train_normal = X_train[y_train == 0]
    #train_groups_normal = train_groups[y_train == 0]
    X_train_normal = X_train
    train_groups_normal = train_groups
    print(f"Train samples: {X_train_normal.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Test blast %: {y_test.mean():.2%}")
    
     # ----- param grid-----
    param_grid = {
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", 0.01, 0.1, 0.5], 
    "nu": [0.001, 0.01, 0.03,0.05, 0.10,0.15,0.2] }
    
    # ======================== INNER CV: HYPERPARAMETER TUNING ========================
    print(f"\n[INNER CV] GridSearchCV hyperparameter tuning...")
    inner_start = time.time()
    
    base_model = OneClassSVM()
    #y_train_normal = y_train[y_train == 0]
    y_train_normal = y_train
    grid_search = GridSearchCV(base_model,param_grid, 
                  cv=inner_cv,scoring=ocsvm_scorer,
                   n_jobs=-1, 
                   verbose=1,refit=True)
    
    grid_search.fit(X_train_normal, y_train_normal, groups=train_groups_normal)
    inner_elapsed = time.time() - inner_start
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Save inner CV results
    inner_cv_df = pd.DataFrame(grid_search.cv_results_)
    inner_cv_df.to_csv(os.path.join(OUTPUT_PATH, "CV_v4_full", f"fold{fold}_inner_cv_results.csv"),index=False)
    
    print(f"[INNER CV] Completed in {inner_elapsed:.1f}s")
    print(f"  Best CV score (ROC-AUC): {best_score:.4f}")
    print(f"  Best params: {best_params}")
    
    # ======================== THRESHOLD OPTIMIZATION ========================
    print(f"\n[THRESHOLD TUNING] Estimating threshold across inner folds...")
    threshold_start = time.time()
    
    fold_thresholds = []
    
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(X_train_normal, y_train_normal, train_groups_normal)):
        
        X_inner_train = X_train_normal[inner_train_idx]
        y_inner_train = y_train_normal.iloc[inner_train_idx]
        X_inner_val = X_train_normal[inner_val_idx]
        y_inner_val = y_train_normal.iloc[inner_val_idx]
        
        # Train model
        model_inner = OneClassSVM(**best_params)
        model_inner.fit(X_inner_train)
        
        # Get decision scores
        scores_inner_val = model_inner.decision_function(X_inner_val)
        
        # Find optimal threshold for this inner fold
        tau_inner, _ = find_optimal_threshold(y_true=y_inner_val.values,y_scores=scores_inner_val,method='f1')
        
        fold_thresholds.append(tau_inner)
        print(f"Inner fold {inner_fold + 1}: threshold = {tau_inner:.4f}")
    
    # Aggregate threshold across inner folds (median for robustness)
    tau_fold = float(np.median(fold_thresholds))
    all_fold_threshold.extend(fold_thresholds)
    threshold_elapsed = time.time() - threshold_start
    
    print(f"[THRESHOLD] Outer fold threshold (median): {tau_fold:.4f}")
    print(f"  Range: [{np.min(fold_thresholds):.4f}, {np.max(fold_thresholds):.4f}]")
    print(f"  Std: {np.std(fold_thresholds):.4f}")
    print(f"  Completed in {threshold_elapsed:.1f}s")
    
     # ======================== OUTER TEST EVALUATION (UNBIASED) ========================
    print(f"\n[OUTER TEST] Evaluating on outer fold test set...")
    
    # Train on FULL outer train set with best params
    #X_train_normal_full = X_train[y_train == 0]
    X_train_normal_full = X_train
    model_final = OneClassSVM(**best_params)
    model_final.fit(X_train_normal_full)
    
    # Get decision scores on outer test set
    scores_test = model_final.decision_function(X_test)
    scores_test_negated = -scores_test
    
    # Apply threshold to get predictions
    y_pred = (scores_test <= tau_fold).astype(int)
    
    # ---- Threshold-INDEPENDENT metrics----
    auprc_test = average_precision_score(y_test, scores_test_negated)
    auroc_test = roc_auc_score(y_test, scores_test_negated)
    
    print(f"\n[METRICS] Threshold-INDEPENDENT:")
    print(f"  AUPRC: {auprc_test:.4f}")
    print(f"  AUROC: {auroc_test:.4f}")
    
    # ---- Threshold-DEPENDENT metrics ----
    prec_test = precision_score(y_test, y_pred, zero_division=0)
    rec_test = recall_score(y_test, y_pred, zero_division=0)
    f1_test = f1_score(y_test, y_pred, zero_division=0)
    
    cm_test = confusion_matrix(y_test, y_pred)
    
    print(f"\n[METRICS] Threshold-DEPENDENT (tau={tau_fold:.4f}):")
    print(f"  Precision: {prec_test:.4f}")
    print(f"  Recall: {rec_test:.4f}")
    print(f"  F1: {f1_test:.4f}")
    print(f"\n[CONFUSION MATRIX] Outer fold {fold}: \n{cm_test}")
    
    print(f"\n[CLASSIFICATION REPORT] Outer fold {fold}:")
    report = classification_report(y_test, y_pred,target_names=["Non-Blast", "Blast"],zero_division=0)
    print(report)
    
    fold_elapsed = time.time() - fold_start
    print(f"\n[TIMING] Outer fold {fold} completed in {fold_elapsed:.1f}s")
    all_outer_fold_results.append({
        'fold': fold,
        'blast_rate_test': blast_rate_test,
        'best_params': best_params,
        'best_score': best_score,
        'inner_thresholds': fold_thresholds,'tau_fold': tau_fold,
        'auprc': auprc_test,'auroc': auroc_test,
        'precision': prec_test,'recall': rec_test,'f1': f1_test,
        'confusion_matrix': cm_test,'classification_report': report,
        'fold_elapsed': fold_elapsed})
    per_fold_visualizations.append({
    'fold': fold,
    'y_test': y_test.values,'y_pred': y_pred,
    'scores_negated': scores_test_negated,
    'tau_fold': tau_fold,
    'auprc': auprc_test,'auroc': auroc_test,
    'precision': prec_test,'recall': rec_test,'f1': f1_test})
    
total_elapsed = time.time() - total_start

print(f"\n{'='*80}")
print(f"NESTED CV COMPLETE")
print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")

# ======================== AGGREGATE RESULTS  ========================
print(f"\n{'='*20}" +"AGGREGATING RESULTS ACROSS ALL OUTER FOLDS"+ "{'='*20}\n")

results_df = pd.DataFrame([{'fold': r['fold'],
        'blast_rate_test': r['blast_rate_test'],
        'auprc': r['auprc'],'auroc': r['auroc'],
        'precision': r['precision'],'recall': r['recall'],'f1': r['f1']}
    for r in all_outer_fold_results])

threshold_summary = pd.DataFrame([{
    'fold': r['fold'], 'precision': r['precision'],
    'recall': r['recall'], 'f1': r['f1']
} for r in all_outer_fold_results])

print("FOLD-BY-FOLD RESULTS:")
print(results_df.to_string(index=False))

# ---- Threshold-INDEPENDENT metrics ----
print(f"\n[THRESHOLD-INDEPENDENT METRICS] (Primary evaluation)")
print(f"  AUPRC: {results_df['auprc'].mean():.4f} +/-{results_df['auprc'].std():.4f}")
print(f"  AUROC: {results_df['auroc'].mean():.4f} +/-{results_df['auroc'].std():.4f}")

# ---- Threshold-DEPENDENT metrics ----
print(f"\n[THRESHOLD-DEPENDENT METRICS] (using per-fold thresholds)")
print(f"  Precision: {results_df['precision'].mean():.4f} +/- {results_df['precision'].std():.4f}")
print(f"  Recall: {results_df['recall'].mean():.4f} +/- {results_df['recall'].std():.4f}")
print(f"  F1: {results_df['f1'].mean():.4f} +/- {results_df['f1'].std():.4f}")

# ---- Threshold statistics ----
final_threshold = np.median(all_fold_threshold)

print(f"\n[THRESHOLD STATISTICS]")
print(f"  Final threshold (median): {final_threshold:.4f}")
print(f"  Range: [{np.min(all_fold_threshold):.4f}, {np.max(all_fold_threshold):.4f}]")

# Save aggregate results
results_df.to_csv(os.path.join(OUTPUT_PATH, "CV_v4_full", "nested_cv_fold_summary.csv"),index=False)

print(f"\n[FILES SAVED]")
print(f"  - nested_cv_fold_summary.csv (aggregate metrics)")
print(f"  - fold*/inner_cv_results.csv (per-fold GridSearch results)")

# ========================================== VISUALIZATION ==========================================
output_dir = os.path.join(OUTPUT_PATH, "plots")
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*80}")
print("[VISUALIZATION] Creating plots for each fold...")
print(f"{'='*80}")

for fold_data in per_fold_visualizations:
    fold = fold_data['fold']
    print(f"\n[INFO] Generating visualizations for Fold {fold + 1}...")
    
    plot_confusion_matrix(fold_data['y_test'], fold_data['y_pred'], fold, output_dir)
    plot_roc_curve(fold_data['y_test'], fold_data['scores_negated'], fold, output_dir)
    plot_precision_recall_curve(fold_data['y_test'], fold_data['scores_negated'], fold, output_dir)
    plot_decision_score_distribution(fold_data['y_test'], fold_data['scores_negated'], fold, output_dir)
    plot_score_by_class_boxplot(fold_data['y_test'], fold_data['scores_negated'], fold, output_dir)
    plot_threshold_line_on_distribution(fold_data['y_test'], fold_data['scores_negated'], 
                                        fold_data['tau_fold'], fold, output_dir)

print(f"\n[INFO] Creating summary metrics plot...")
plot_summary_metrics(per_fold_visualizations, output_dir)

print(f"\n{'='*80}")
print("[SUCCESS] All visualizations generated!")
print(f"{'='*80}")
print(f"\nOutput directory: {output_dir}")

# ========================================== STEP 4: FINAL MODEL SAVING ==========================================

import pickle

print("\n" + "="*30 + " FINAL MODEL SAVING " + "="*30)

# Find best fold (highest F1)
best_fold = max(all_outer_fold_results, key=lambda x: x["f1"])
best_params = best_fold["best_params"]
tau_final = best_fold["tau_fold"]

print(f"\n[FINAL] Best fold: {best_fold['fold'] + 1}")
print(f"[FINAL] Best F1: {best_fold['f1']:.4f}")

# Train on ALL normal samples
X_all_normal = X_all[y == 0]
print(f"\n[TRAINING] Fitting OneClassSVM on {X_all_normal.shape[0]} normal samples...")

final_ocsvm = OneClassSVM(**best_params)
final_ocsvm.fit(X_all_normal)

print(f"[SUCCESS] Model trained! Support vectors: {final_ocsvm.n_support_}")

# Save everything
model_artifacts = {
    "model": final_ocsvm,
    "threshold": tau_final,
    "features": features,
    "best_params": best_params,
    "metadata": {
        "auprc": best_fold['auprc'], "auroc": best_fold['auroc'],
        "f1": best_fold['f1'], "precision": best_fold['precision'], "recall": best_fold['recall'],
        "n_support_vectors": int(final_ocsvm.n_support_)}}

save_path = os.path.join(OUTPUT_PATH, "final_ocsvm_model_with_threshold.pkl")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'wb') as file:
    pickle.dump(model_artifacts, file)

print(f"[SUCCESS] Model saved: {save_path}")

# Save metadata
metadata_df = pd.DataFrame([model_artifacts["metadata"]])
metadata_path = os.path.join(OUTPUT_PATH, "final_model_metadata.csv")
metadata_df.to_csv(metadata_path, index=False)

print(f"[SUCCESS] Metadata saved: {metadata_path}")

#===================================================================================
script_end_time = time.time()
total_seconds = int(script_end_time - script_start_time)
hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60
runtime_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETE")
print("="*80)
print(f"Total Runtime: {runtime_formatted} (hh:mm:ss)")
print(f"Total Seconds: {total_seconds}")
print("="*80 + "\n")