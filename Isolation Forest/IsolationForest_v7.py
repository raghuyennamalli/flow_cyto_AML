import os
import time
import numpy as np
import pandas as pd
import re  
from FlowCytometryTools import FCMeasurement
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer,
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, auc)
from joblib import parallel_backend
import matplotlib.pyplot as plt
import pickle
from sklearn.base import clone
import seaborn as sns
from collections import Counter

# ========================================= PARAMETERS & PATHS ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/Changed_para/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "IF_CV"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "IF_Plots"), exist_ok=True)

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
    """Score-based AUPRC for IF hyperparameter tuning (continuous scores).
    - decision_function: higher = more normal (WBC), lower = anomalous (Blast) --> For AUPRC: we need y_true=1 → high score
    - Negation converts: -score means blasts (y=1) get higher negated scores"""
    y_scores_if = estimator.decision_function(X)
    y_scores_negated = -y_scores_if 
    return average_precision_score(y, y_scores_negated)
    
def optimal_threshold_f1(y_true, y_scores_inverted):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
    thresholds = np.append(thresholds, 0)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    return -thresholds[ix], precision, recall


def find_optimal_threshold(y_true, y_scores, method='clinical_recall'):
    """Find optimal threshold for Isolation Forest decision function.- We want: scores <= threshold → classify as blast (class 1)
    Methods:
    - 'clinical_recall': Maximize recall while ensuring precision >= 0.90
    - 'f1': Maximize F1 score (balanced precision/recall)
    """
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
    
    return optimal_thresh, {
        'threshold': optimal_thresh,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)}

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
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Blast', 'Blast'], yticklabels=['Non-Blast', 'Blast'], cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12)
    save_fig(os.path.join(output_path, f'fold{fold}_confusion_matrix.png'))

def plot_roc_curve(y_true, y_scores, fold, output_path): #ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
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
    

def plot_precision_recall_curve(y_true, y_scores, fold, output_path):     #Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
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

def plot_anomaly_score_distribution(y_true, y_scores, fold, output_path):      #Plot distribution of anomaly scores
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label=f'Non-blasts (n={len(y_scores[y_true == 0])})', color='blue', edgecolor='black')
    plt.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label=f'Blast (n={len(y_scores[y_true == 1])})', color='red', edgecolor='black')
    plt.xlabel('Anomaly Score', fontsize=12); plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Anomaly Score Distribution - Fold {fold+1}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11); plt.grid(alpha=0.3, axis='y')
    save_fig(os.path.join(output_path, f'fold{fold}_anomaly_score_distribution.png'))

def plot_score_by_class_boxplot(y_true, y_scores, fold, output_path):     #Plot box plot of scores by class
    df_scores = pd.DataFrame({'Score': y_scores,'Class': ['Non-blasts' if y == 0 else 'Blast' for y in y_true]})
    
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
        metric_name = m.replace('outer_', '').capitalize()  
        ax.bar(folds, df[m], color=c, edgecolor='black', alpha=0.7)
        ax.set_title(f'{m.capitalize()} Across Folds', fontweight='bold'); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate(df[m]):
            ax.text(folds.iloc[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
            
    means = df[metrics].mean()
    stds = df[metrics].std()

    axes[1, 1].bar(metrics, means, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Average Metrics ± Std', fontweight='bold'); axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(axis='y', alpha=0.3)
    save_fig(os.path.join(output_path, 'summary_metrics.png'))

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

    make_5pct_dataset(data=data_2K,output_path=os.path.join(OUTPUT_PATH, "BLAST110_2K_5pct.pkl"))
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

X_all= X.values

print(f"[INFO] Data shape: {X_all.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ========================================== STEP 2: DEFINE INNER + OUTER CV ==========================================
inner_cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
outer_cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)

from sklearn.base import clone

# ========================================== STEP 3: OUTER CV LOOP ==========================================
total_start_time = time.time()
all_outer_fold_results = []
all_inner_thresholds = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y, groups=groups)):
    print(f"\n{'='*30} OUTER FOLD {fold+1}/{outer_cv.get_n_splits()} {'='*30}")
    fold_start_time = time.time()

    # ----- split data for this outer fold -----
    X_train = X_all[train_idx]
    y_train = y.iloc[train_idx]
    X_test  = X_all[test_idx]
    y_test  = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    blast_prevalence = y_train.mean()
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")
    print(f"Train blast %: {y_train.mean():.2%}, Test blast %: {y_test.mean():.2%}")

    # ----- param grid (depends on outer-train prevalence) -----
    contamination_range = [blast_prevalence - 0.03, 0.01,
        blast_prevalence - 0.015, blast_prevalence,
        min(0.5, blast_prevalence + 0.03) ]
    contamination_range = [max(0.001, c) for c in contamination_range]
    param_grid = {"n_estimators": [100, 200, 300, 400],
        "max_samples": [64, 128, 256],
        "max_features": [0.44, 0.5, 0.75, 1.0],
        "contamination": contamination_range}
    print(f"[INFO] Contamination range: {[f'{c:.4f}' for c in contamination_range]}")

    # ------------------------ INNER CV (hyperparameter tuning) ------------------------
    print(f"\n[INNER CV] Starting grid search with parallelization...")
    inner_cv_start = time.time()

    base_iso = IsolationForest(random_state=fold, n_jobs=1)
    grid = GridSearchCV(base_iso, param_grid, cv=inner_cv,
        scoring=iforest_auprc_continuous,
        n_jobs=5,
        refit=True,verbose=0,)
    start = time.time()
    grid.fit(X_train, y_train, groups=train_groups)
    elapsed_inner = time.time() - start
    inner_cv_elapsed = time.time() - inner_cv_start

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(os.path.join(OUTPUT_PATH, "IF_CV", f"fold{fold}_innerCV.csv"), index=False)

    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best AUPRC: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")

    # ------------------------ threshold tuning ------------------------
    outer_test_start = time.time()
    print(f"\n[INNER CV] Threshold tuning...")

    best_iso = grid.best_estimator_
    inner_thresholds = []

    for inner_fold, (inner_tr_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train, groups=train_groups)):
        X_tr_in  = X_train[inner_tr_idx]
        y_tr_in  = y_train.iloc[inner_tr_idx]
        X_val_in = X_train[inner_val_idx]
        y_val_in = y_train.iloc[inner_val_idx]

        iso_inner = clone(best_iso)
        iso_inner.fit(X_tr_in, y_tr_in)

        y_scores_val_if = iso_inner.decision_function(X_val_in)

        tau_inner, _ = find_optimal_threshold(y_true=y_val_in.values,y_scores=y_scores_val_if,method="f1")
        inner_thresholds.append(tau_inner)

    tau_fold = float(np.median(inner_thresholds))   
    all_inner_thresholds.append({"fold": fold, "inner_thresholds": inner_thresholds, "tau_fold": tau_fold})
    print(f"[INNER CV] Fold {fold+1} threshold (median of inner folds): {tau_fold:.4f}")

    # ======================== OUTER TESTING (UNBIASED) ========================
    print(f"\n[OUTER TEST] Evaluating on outer test fold (no tuning here)...")

    best_iso = clone(best_iso)
    best_iso.fit(X_train, y_train)

    y_scores_test_if = best_iso.decision_function(X_test)
    y_scores_test_negated = -y_scores_test_if
    
    y_pred_test = ( y_scores_test_if <= tau_fold).astype(int)

    print(f"  Collected {len(y_scores_test_if):,} test samples")
    print(f"  Test blast prevalence: {y_test.mean():.3%}")
    print(f"  Score range: [{np.min(y_scores_test_if):.4f}, {np.max(y_scores_test_if):.4f}]")

    # Threshold-independent metrics
    outer_auprc = average_precision_score(y_test, y_scores_test_negated)
    outer_auroc = roc_auc_score(y_test, y_scores_test_negated)

    report = classification_report(y_test,y_pred_test,target_names=["Non-Blast", "Blast"],output_dict=True,zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    print(f"\n[OUTER CV] Fold {fold+1} - THRESHOLD-INDEPENDENT Metrics (PRIMARY):")
    print(f"  AUPRC: {outer_auprc:.4f}")
    print(f"  AUROC: {outer_auroc:.4f}")

    print(f"\n[OUTER CV] Fold {fold+1} - THRESHOLD-DEPENDENT Metrics (inner-CV threshold):")
    print(f"  Threshold: {tau_fold:.4f}\n{report}")
    print(f"  Confusion Matrix: \n{cm}")

    outer_test_elapsed = time.time() - outer_test_start

    # ======================== VISUALIZATIONS ========================
    print(f"\n[VISUALIZATION] Creating plots...")
    plot_confusion_matrix(y_test, y_pred_test, fold,os.path.join(OUTPUT_PATH, "IF_Plots"))
    plot_roc_curve(y_test, y_scores_test_negated, fold, os.path.join(OUTPUT_PATH, "IF_Plots"))
    plot_precision_recall_curve(y_test, y_scores_test_negated, fold, os.path.join(OUTPUT_PATH, "IF_Plots"))
    plot_anomaly_score_distribution(y_test, y_scores_test_if, fold, os.path.join(OUTPUT_PATH, "IF_Plots"))
    plot_score_by_class_boxplot(y_test, y_scores_test_if, fold, os.path.join(OUTPUT_PATH, "IF_Plots"))

    fold_elapsed = time.time() - fold_start_time

    NonBlast = report["Non-Blast"]; blast = report["Blast"]
    
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)  
    rec = recall_score(y_test,y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)
    
    

    all_outer_fold_results.append({"fold": fold,
        "blast_prevalence_train": blast_prevalence,
        "outer_auprc": outer_auprc,
        "outer_auroc": outer_auroc,
        "tau_fold": tau_fold,
        "best_params": grid.best_params_,
        "precision": prec, "accuracy":acc, "recall": rec, "f1": f1,
        "precision_nonBlast": NonBlast["precision"],"recall_nonBlast": NonBlast["recall"],"f1_nonBlast": NonBlast["f1-score"],
        "precision_blast": blast["precision"],"recall_blast": blast["recall"],"f1_blast": blast["f1-score"],
        "n_test_samples": len(y_test), "n_test_blasts": int(np.sum(y_test.values)),
        "test_blast_prevalence": y_test.mean(),
        "inner_cv_time": inner_cv_elapsed, "outer_test_time": outer_test_elapsed,
        "total_fold_time": fold_elapsed })

    print(f"\n[FOLD TIMING]")
    print(f"  - Inner CV: {inner_cv_elapsed:.1f}s")
    print(f"  - Outer test: {outer_test_elapsed:.1f}s")
    print(f"  - Total fold time: {fold_elapsed:.1f}s")

#========================================== STEP 4: FINAL MODEL SAVING ==========================================
print("\n" + "="*30 + " FINAL MODEL SAVING " + "="*30)

all_best_params = [res["best_params"] for res in all_outer_fold_results]
param_tuples = [tuple(sorted(p.items())) for p in all_best_params]
best_param_tuple = Counter(param_tuples).most_common(1)[0][0]
best_params = dict(best_param_tuple)

all_tau_folds = [res["tau_fold"] for res in all_outer_fold_results]
tau_final = float(np.median(all_tau_folds))
print(f"[FINAL] Selected params: {best_params} \n[FINAL] Selected threshold: {tau_final:.4f}")

final_iso = IsolationForest(**best_params,random_state=42,n_jobs=-1)
final_iso.fit(X_all)

model_artifacts = {"model": final_iso, "threshold": tau_final, "features": features}

save_path = os.path.join(OUTPUT_PATH, "final_if_model_with_threshold.pkl")
with open(save_path, 'wb') as file:
    pickle.dump(model_artifacts, file)

print(f"[INFO] Final model + threshold saved at: {save_path}")

#========================================== FINAL SUMMARY ==========================================
print("\n" + "="*30 + " NESTED CV FINAL SUMMARY " + "="*30)

# Create summary dataframe from all folds
results_df = pd.DataFrame(all_outer_fold_results)

print(f"\n[HYPERPARAMETER TUNING] Results across {len(all_outer_fold_results)} outer folds:")
print(f"\n{results_df[['fold', 'outer_auprc', 'outer_auroc', 'test_blast_prevalence']].to_string(index=False)}")

print(f"\n[METRICS SUMMARY]")
print(f"  AUPRC: {results_df['outer_auprc'].mean():.4f} ± {results_df['outer_auprc'].std():.4f}")
print(f"  AUROC: {results_df['outer_auroc'].mean():.4f} ± {results_df['outer_auroc'].std():.4f}")
print(f"  Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"  Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
print(f"  Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
print(f"  F1-Score: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
print(f"  Test prevalence: {results_df['test_blast_prevalence'].mean():.3%} ± {results_df['test_blast_prevalence'].std():.3%}")
results_df.to_csv(os.path.join(OUTPUT_PATH,"IF_CV", 'nested_cv_summary.csv'), index=False)
print(f"\n[INFO] Summary saved to {os.path.join(OUTPUT_PATH, 'nested_cv_summary.csv')}")
print(f"\n{'='*80} ANALYSIS COMPLETE {'='*80}\n")


# ========================================== FINAL RUNTIME REPORT ==========================================
total_elapsed = time.time() - total_start_time

print(f"\n[RUNTIME SUMMARY]")
print(f"Total runtime: {total_elapsed:.1f} seconds")
print(f"             = {total_elapsed/60:.1f} minutes")
print(f"             = {total_elapsed/3600:.2f} hours")
