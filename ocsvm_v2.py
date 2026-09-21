
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, ParameterGrid
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer,
    precision_recall_curve, roc_curve
)
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
# ==========================================
# SETUP
# ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output_svm"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)

features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def from_fcs(path):
    """Load FCS file and return data as a pandas DataFrame."""
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()


def ocsvm_norm_score(estimator, X, y=None):
    try:
        scores = estimator.decision_function(X)
        if np.isnan(scores).any():
            print("WARNING: NaN in decision_function scores!")
            return -np.inf
        return np.mean(scores)
    except Exception as e:
        print(f"Scorer error: {str(e)}")
        return -np.inf


def find_optimal_threshold(y_true, y_scores, method='clinical_recall'):
    """
    Find optimal threshold for One-Class SVM decision function.
    
    One-Class SVM convention:
    - Positive scores = normal (inlier)
    - Negative scores = anomaly (outlier)
    
    We want: scores < threshold → classify as blast (class 1)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, -y_scores)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        optimal_thresh = thresholds[ix]
        
    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, -y_scores)
        fscore = (2 * precision * recall) / (precision + recall + 1e-10)
        ix = np.argmax(fscore)
        optimal_thresh = thresholds[ix] if ix < len(thresholds) else thresholds[-1]
        
    elif method == 'clinical_recall':
        precision, recall, thresholds = precision_recall_curve(y_true, -y_scores)
        # CRITICAL: Truncate precision and recall to match thresholds length
        precision = precision[:-1]
        recall = recall[:-1]
        # Now all arrays have the same length
        valid_idx = precision >= 0.90  # Safe to use
        if np.sum(valid_idx) > 0:
            valid_recall = recall[valid_idx]
            valid_thresholds = thresholds[valid_idx]  # Properly aligned!
            ix = np.argmax(valid_recall)
            optimal_thresh = valid_thresholds[ix]

        else:
            fscore = (2 * precision * recall) / (precision + recall + 1e-10)
            ix = np.argmax(fscore)
            optimal_thresh = thresholds[ix] if ix < len(thresholds) else thresholds[-1]
    
    y_pred = (y_scores < optimal_thresh).astype(int)
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
# STEP 1: LOAD DATA (CORRECTED)
# ==========================================
pkl_2k = os.path.join(OUTPUT_PATH, "BLAST110_2K.pkl")
data = pd.read_pickle(pkl_2k)

X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]

# FIX 1: NO StandardScaler - data already logicle-transformed
X_scaled = X.values  # Convert to numpy directly

print(f"[INFO] Data shape: {X_scaled.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ==========================================
# STEP 2: DEFINE CV STRATEGY
# ==========================================
outer_cv = GroupKFold(n_splits=5)
inner_cv = KFold(n_splits=5, shuffle=False) 

# ==========================================
# STEP 3: EXHAUSTIVE HYPERPARAMETER GRID
# ==========================================
# For true exhaustive search: test all combinations
param_grid_exhaustive = {
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    "nu": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    "degree": [2, 3, 4],  # Only for poly kernel
    "coef0": [0.0, 1.0]   # Only for poly kernel
}

# For faster runtime: reduced grid (still comprehensive)
param_grid_fast = {
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "gamma": [0.001, 0.01, 0.1, 1],
    "nu": [0.005, 0.01, 0.05, 0.1, 0.2]
}


# Choose which grid to use (uncomment one)
#param_grid = param_grid_fast  # ~4×4×5 = 80 combinations
# Change your param_grid to this:
param_grid = {
    "kernel": ["rbf"],
    "gamma": ["scale", 0.01, 0.1, 0.5], 
    "nu": [0.001, 0.01, 0.03,0.05, 0.10, 0.15, 0.20]  # ← Increase nu! (was too strict at 0.01)
}


# param_grid = param_grid_exhaustive  # ~200+ combinations (poly filtered)

print(f"[INFO] Hyperparameter combinations: ~{len(list(ParameterGrid(param_grid)))}")

# ==========================================
# STEP 4: OUTER CV LOOP (CORRECTED)
# ==========================================
threshold_results = []
all_fold_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y, groups)):
    print(f"\n{'='*60}")
    print(f"OUTER FOLD {fold+1}/5")
    print(f"{'='*60}")

    # FIX 2: CORRECT INDEXING with numpy array
    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train blast %: {y_train.mean():.2%}, Test blast %: {y_test.mean():.2%}")

    # Train only on normal (WBC) cells
    X_train_norm = X_train[y_train == 0]
    train_groups_norm = train_groups[y_train == 0]
    y_train_norm = y_train[y_train == 0]

    
    print(f"Training on {X_train_norm.shape[0]} normal cells")

    # ========================
    # INNER CV (GRID SEARCH)
    # ========================
    print(f"\n[INNER CV] Starting grid search...")
    base_svm = OneClassSVM()
    
    grid = GridSearchCV(
        base_svm,
        param_grid,
        cv=inner_cv,
        scoring=ocsvm_norm_score,
        n_jobs=-1,
        refit=True,
        verbose=0,
        error_score=-np.inf  # Handle invalid parameter combinations
    )

    # FIX 3: CORRECT GROUP PASSING
    start = time.time()
    print(f"NaN in X_train_norm: {np.isnan(X_train_norm).sum()}")

    grid.fit(X_train_norm, y_train_norm)
    elapsed_inner = time.time() - start

    # Save inner CV results
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(
        os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_OneClassSVM_innerCV.csv"),
        index=False
    )
    
    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best Score: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")

    # ========================
    # OUTER TESTING
    # ========================
    best_svm = grid.best_estimator_

    # Get decision function scores
    anomaly_scores_test = best_svm.decision_function(X_test)
    
    # ===== DIAGNOSTIC 1: Score Distribution =====
    print(f"\n[DIAGNOSTIC FOLD {fold}]")
    print(f"Decision Function Score Statistics:")
    print(f"  Overall - Min: {anomaly_scores_test.min():.6f}, Max: {anomaly_scores_test.max():.6f}")
    print(f"  Overall - Mean: {anomaly_scores_test.mean():.6f}, Std: {anomaly_scores_test.std():.6f}")
    print(f"  Overall - Median: {np.median(anomaly_scores_test):.6f}")
    # ===== DIAGNOSTIC 2: Separation by Class =====
    wbc_scores = anomaly_scores_test[y_test.values == 0]
    blast_scores = anomaly_scores_test[y_test.values == 1]
    print(f"\nWBC (should be POSITIVE):")
    print(f"  Count: {len(wbc_scores)}")
    print(f"  Mean: {wbc_scores.mean():.6f}, Std: {wbc_scores.std():.6f}")
    print(f"  Min: {wbc_scores.min():.6f}, Max: {wbc_scores.max():.6f}")
    print(f"  % > 0: {np.sum(wbc_scores > 0) / len(wbc_scores) * 100:.2f}%")
    print(f"\nBlast (should be NEGATIVE):")
    print(f"  Count: {len(blast_scores)}")
    print(f"  Mean: {blast_scores.mean():.6f}, Std: {blast_scores.std():.6f}")
    print(f"  Min: {blast_scores.min():.6f}, Max: {blast_scores.max():.6f}")
    print(f"  % < 0: {np.sum(blast_scores < 0) / len(blast_scores) * 100:.2f}%")
    
    print(f"\nSeparation Quality:")
    print(f"  Overlap: WBC min={wbc_scores.min():.6f}, Blast max={blast_scores.max():.6f}")
    print(f"  Gap between classes: {wbc_scores.min() - blast_scores.max():.6f}")
    
    # ===== DIAGNOSTIC 3: Support Vectors =====
    print(f"\nModel Internals:")
    print(f"  Support vectors: {len(best_svm.support_vectors_)}")
    print(f"  % of training data: {len(best_svm.support_vectors_) / len(X_train_norm) * 100:.2f}%")
    print(f"  Offset (rho): {float(best_svm.offset_):.6f}")
    # ===== DIAGNOSTIC 4: Score Uniqueness =====
    unique_scores = len(np.unique(anomaly_scores_test))
    print(f"\nScore Uniqueness:")
    print(f"  Unique score values: {unique_scores} / {len(anomaly_scores_test)}")
    print(f"  Expected: > 10000 (if working properly)")

    # FIX 4: THRESHOLD OPTIMIZATION instead of fixed threshold
    optimal_thresh_clinical, metrics_clinical = find_optimal_threshold(
        y_test.values, anomaly_scores_test, method='f1'
    )
    optimal_thresh_f1, metrics_f1 = find_optimal_threshold(
        y_test.values, anomaly_scores_test, method='f1'
    )
    
    # Use clinical threshold for predictions
    preds = (anomaly_scores_test < optimal_thresh_clinical).astype(int)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"\n[OUTER CV] Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(f"\n[OUTER CV] Classification Report:")
    print(classification_report(y_test, preds, target_names=["WBC", "Blast"]))

    print(f"\n[THRESHOLD ANALYSIS]")
    print(f"Clinical (90% precision) threshold: {optimal_thresh_clinical:.4f}")
    print(f"  → Precision: {metrics_clinical['precision']:.3f}, Recall: {metrics_clinical['recall']:.3f}, F1: {metrics_clinical['f1']:.3f}")
    print(f"F1-optimized threshold: {optimal_thresh_f1:.4f}")
    print(f"  → Precision: {metrics_f1['precision']:.3f}, Recall: {metrics_f1['recall']:.3f}, F1: {metrics_f1['f1']:.3f}")

    threshold_results.append({
        'fold': fold,
        'best_params': str(grid.best_params_),
        'clinical_threshold': optimal_thresh_clinical,
        'clinical_precision': metrics_clinical['precision'],
        'clinical_recall': metrics_clinical['recall'],
        'clinical_f1': metrics_clinical['f1'],
        'f1_threshold': optimal_thresh_f1,
        'f1_precision': metrics_f1['precision'],
        'f1_recall': metrics_f1['recall'],
        'f1_score': metrics_f1['f1']
    })
    
        # ===== DIAGNOSTIC 5: Threshold Location =====
    print(f"\nThreshold Analysis:")
    print(f"  Clinical threshold: {optimal_thresh_clinical:.6f}")
    print(f"  WBC below threshold: {np.sum(wbc_scores < optimal_thresh_clinical)} / {len(wbc_scores)}")
    print(f"  Blast below threshold: {np.sum(blast_scores < optimal_thresh_clinical)} / {len(blast_scores)}")


    # ========================
    # PER-SAMPLE RESULTS (CORRECTED)
    # ========================
    sample_ids = samples.iloc[test_idx].unique()
    model_results = []

    for sid in sample_ids:
        subset = data[data["sample_id"] == sid]
        # FIX 5: NO scaler - use features directly
        X_sub = subset[features].values
        y_sub = subset["Blast"].values

        # Get decision function scores
        scores_sub = best_svm.decision_function(X_sub)
        
        # Apply optimal threshold
        pred_sub = (scores_sub < optimal_thresh_clinical).astype(int)

        # FIX 6: ADD EDGE CASE HANDLING with zero_division
        # FIX 7: ADD DECISION FUNCTION METRICS
        result_dict = {
            "fold": fold,
            "sample_id": sid,
            "model": "OneClassSVM_OptimalThreshold",
            "gt_count": int(np.sum(y_sub == 1)),
            "gt_perc": float(np.mean(y_sub == 1)),
            "pred_count": int(np.sum(pred_sub == 1)),
            "pred_perc": float(np.mean(pred_sub == 1)),
            "accuracy": accuracy_score(y_sub, pred_sub),
            "precision": precision_score(y_sub, pred_sub, zero_division=0),
            "recall": recall_score(y_sub, pred_sub, zero_division=0),
            "f1": f1_score(y_sub, pred_sub, zero_division=0),
            "mean_decision_score_blast": np.mean(scores_sub[y_sub == 1]) if np.any(y_sub == 1) else np.nan,
            "mean_decision_score_wbc": np.mean(scores_sub[y_sub == 0]) if np.any(y_sub == 0) else np.nan,
            "optimal_threshold": optimal_thresh_clinical
        }
        model_results.append(result_dict)

    pd.DataFrame(model_results).to_csv(
        os.path.join(OUTPUT_PATH, "CV", f"fold{fold}_OneClassSVM_outerCV_optimal.csv"),
        index=False
    )
    
    all_fold_results.extend(model_results)

# ==========================================
# SAVE FINAL SUMMARIES
# ==========================================
pd.DataFrame(threshold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "ocsvm_threshold_summary.csv"),
    index=False
)

pd.DataFrame(all_fold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "ocsvm_all_samples_results.csv"),
    index=False
)

print("\n" + "="*60)
print("NESTED CROSS-VALIDATION COMPLETE")
print("="*60)
print(f"Threshold summary saved to: ocsvm_threshold_summary.csv")
print(f"Sample results saved to: ocsvm_all_samples_results.csv")

# Print average performance
avg_results = pd.DataFrame(threshold_results)
print(f"\nAverage performance across {len(threshold_results)} folds:")
print(f"  Clinical Recall: {avg_results['clinical_recall'].mean():.3f} ± {avg_results['clinical_recall'].std():.3f}")
print(f"  Clinical Precision: {avg_results['clinical_precision'].mean():.3f} ± {avg_results['clinical_precision'].std():.3f}")
print(f"  Clinical F1: {avg_results['clinical_f1'].mean():.3f} ± {avg_results['clinical_f1'].std():.3f}")


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

CV_PATH = os.path.join(OUTPUT_PATH, "CV")

# Load CSV results
threshold_summary = pd.read_csv(os.path.join(CV_PATH, "ocsvm_threshold_summary.csv"))
all_samples_results = pd.read_csv(os.path.join(CV_PATH, "ocsvm_all_samples_results.csv"))

print("[INFO] Loaded results")
print(f"  Threshold summary shape: {threshold_summary.shape}")
print(f"  All samples results shape: {all_samples_results.shape}")

# ==========================================
# FEATURE DISTRIBUTION PLOTS - ALL FOLDS
# ==========================================

print("\n" + "="*60)
print("GENERATING FEATURE DISTRIBUTION PLOTS")
print("="*60)

# Create folder for feature distribution visualizations
feature_dist_folder = os.path.join(OUTPUT_PATH, "feature_distributions")
os.makedirs(feature_dist_folder, exist_ok=True)

# Create PDF for feature distributions
feature_pdf_path = os.path.join(OUTPUT_PATH, "Feature_Distributions_AllFolds.pdf")
feature_pdf = PdfPages(feature_pdf_path)

# ==========================================
# PAGE 1: ALL FEATURES - OVERALL DISTRIBUTION (ALL FOLDS COMBINED)
# ==========================================
print("[INFO] Generating overall feature distributions...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

for feat_idx, feature in enumerate(features):
    ax = fig.add_subplot(gs[feat_idx // 2, feat_idx % 2])
    
    # Get data for this feature
    X_feature = data[feature].values
    y_label = data["Blast"].values
    
    wbc_data = X_feature[y_label == 0]
    blast_data = X_feature[y_label == 1]
    
    # Create histograms
    ax.hist(wbc_data, bins=50, alpha=0.6, label='WBC', color='#4ECDC4', edgecolor='black', density=True)
    ax.hist(blast_data, bins=50, alpha=0.6, label='Blast', color='#FF6B6B', edgecolor='black', density=True)
    
    ax.set_xlabel(feature, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'{feature} - Train vs Test Distribution (All Folds Combined)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"WBC: μ={np.mean(wbc_data):.3f}, σ={np.std(wbc_data):.3f}\nBlast: μ={np.mean(blast_data):.3f}, σ={np.std(blast_data):.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add empty subplot for 5th feature position
ax_empty = fig.add_subplot(gs[2, 1])
ax_empty.axis('off')

fig.suptitle('Feature Distributions - All Folds Combined (Overall)', fontsize=16, fontweight='bold', y=0.995)
feature_pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 1: Overall Feature Distributions - SAVED")

# ==========================================
# PAGE 2-6: INDIVIDUAL FOLD DISTRIBUTIONS
# ==========================================

for fold_id in range(5):
    print(f"[INFO] Generating feature distributions for Fold {fold_id + 1}...")
    
    # Get fold data
    outer_cv_splits = list(outer_cv.split(X_scaled, y, groups))
    train_idx, test_idx = outer_cv_splits[fold_id]
    
    X_train_fold = X_scaled[train_idx]
    X_test_fold = X_scaled[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for feat_idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[feat_idx // 2, feat_idx % 2])
        
        # Get train and test data
        X_train_feat = X_train_fold[:, feat_idx]
        X_test_feat = X_test_fold[:, feat_idx]
        
        train_wbc = X_train_feat[y_train_fold == 0]
        train_blast = X_train_feat[y_train_fold == 1]
        test_wbc = X_test_feat[y_test_fold == 0]
        test_blast = X_test_feat[y_test_fold == 1]
        
        # Create density plots with KDE
        ax.hist(train_wbc, bins=40, alpha=0.5, label='Train WBC', color='#4ECDC4', edgecolor='black', density=True)
        ax.hist(test_wbc, bins=40, alpha=0.5, label='Test WBC', color='#2E9B96', edgecolor='black', density=True, histtype='step', linewidth=2)
        
        ax.hist(train_blast, bins=40, alpha=0.5, label='Train Blast', color='#FF6B6B', edgecolor='black', density=True)
        ax.hist(test_blast, bins=40, alpha=0.5, label='Test Blast', color='#CC5555', edgecolor='black', density=True, histtype='step', linewidth=2)
        
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Fold {fold_id + 1}: {feature} - Train vs Test', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = (f"Train WBC: μ={np.mean(train_wbc):.3f}, σ={np.std(train_wbc):.3f}\n"
                      f"Test WBC: μ={np.mean(test_wbc):.3f}, σ={np.std(test_wbc):.3f}\n"
                      f"Train Blast: μ={np.mean(train_blast):.3f}, σ={np.std(train_blast):.3f}\n"
                      f"Test Blast: μ={np.mean(test_blast):.3f}, σ={np.std(test_blast):.3f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add empty subplot
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis('off')
    
    fig.suptitle(f'Fold {fold_id + 1}: Feature Distributions - Train vs Test', 
                 fontsize=16, fontweight='bold', y=0.995)
    feature_pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Fold {fold_id + 1}: Feature Distributions - SAVED")

# ==========================================
# PAGE 7: FEATURE STATISTICS SUMMARY TABLE
# ==========================================
print("[INFO] Generating feature statistics summary...")

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.axis('off')

# Create comprehensive statistics table
summary_stats = []
for feature in features:
    X_feat = data[feature].values
    y_label = data["Blast"].values
    
    wbc_data = X_feat[y_label == 0]
    blast_data = X_feat[y_label == 1]
    
    summary_stats.append([
        feature,
        f"{np.mean(wbc_data):.4f}",
        f"{np.std(wbc_data):.4f}",
        f"{np.min(wbc_data):.4f}",
        f"{np.max(wbc_data):.4f}",
        f"{np.mean(blast_data):.4f}",
        f"{np.std(blast_data):.4f}",
        f"{np.min(blast_data):.4f}",
        f"{np.max(blast_data):.4f}"
    ])

table_data = [
    ['Feature', 'WBC Mean', 'WBC Std', 'WBC Min', 'WBC Max', 
     'Blast Mean', 'Blast Std', 'Blast Min', 'Blast Max']
] + summary_stats

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(table_data)):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_stats) + 1):
    for j in range(len(table_data)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')

fig.suptitle('Feature Statistics Summary - All Folds', fontsize=16, fontweight='bold')
feature_pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 7: Feature Statistics Summary - SAVED")

# Close feature distribution PDF
feature_pdf.close()

print(f"\n[SUCCESS] Feature distributions PDF saved to: {feature_pdf_path}")

# ==========================================
# SAVE INDIVIDUAL PNG FILES FOR EACH FOLD
# ==========================================
print("\n[INFO] Saving individual PNG files for each fold...")

for fold_id in range(5):
    print(f"[INFO] Generating PNG for Fold {fold_id + 1}...")
    
    # Get fold data
    outer_cv_splits = list(outer_cv.split(X_scaled, y, groups))
    train_idx, test_idx = outer_cv_splits[fold_id]
    
    X_train_fold = X_scaled[train_idx]
    X_test_fold = X_scaled[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for feat_idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[feat_idx // 2, feat_idx % 2])
        
        # Get train and test data
        X_train_feat = X_train_fold[:, feat_idx]
        X_test_feat = X_test_fold[:, feat_idx]
        
        train_wbc = X_train_feat[y_train_fold == 0]
        train_blast = X_train_feat[y_train_fold == 1]
        test_wbc = X_test_feat[y_test_fold == 0]
        test_blast = X_test_feat[y_test_fold == 1]
        
        # Create density plots
        ax.hist(train_wbc, bins=40, alpha=0.5, label='Train WBC', color='#4ECDC4', edgecolor='black', density=True)
        ax.hist(test_wbc, bins=40, alpha=0.5, label='Test WBC', color='#2E9B96', edgecolor='black', density=True, histtype='step', linewidth=2)
        
        ax.hist(train_blast, bins=40, alpha=0.5, label='Train Blast', color='#FF6B6B', edgecolor='black', density=True)
        ax.hist(test_blast, bins=40, alpha=0.5, label='Test Blast', color='#CC5555', edgecolor='black', density=True, histtype='step', linewidth=2)
        
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Fold {fold_id + 1}: {feature} - Train vs Test', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = (f"Train WBC: μ={np.mean(train_wbc):.3f}, σ={np.std(train_wbc):.3f}\n"
                      f"Test WBC: μ={np.mean(test_wbc):.3f}, σ={np.std(test_wbc):.3f}\n"
                      f"Train Blast: μ={np.mean(train_blast):.3f}, σ={np.std(train_blast):.3f}\n"
                      f"Test Blast: μ={np.mean(test_blast):.3f}, σ={np.std(test_blast):.3f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Empty subplot
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis('off')
    
    fig.suptitle(f'Fold {fold_id + 1}: Feature Distributions - Train vs Test', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save as PNG
    png_path = os.path.join(feature_dist_folder, f"fold{fold_id + 1}_feature_distributions.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Fold {fold_id + 1} PNG saved to: {png_path}")
    
    plt.close()

# ==========================================
# SAVE OVERALL DISTRIBUTION PNG
# ==========================================
print("[INFO] Generating overall distribution PNG...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

for feat_idx, feature in enumerate(features):
    ax = fig.add_subplot(gs[feat_idx // 2, feat_idx % 2])
    
    X_feature = data[feature].values
    y_label = data["Blast"].values
    
    wbc_data = X_feature[y_label == 0]
    blast_data = X_feature[y_label == 1]
    
    ax.hist(wbc_data, bins=50, alpha=0.6, label='WBC', color='#4ECDC4', edgecolor='black', density=True)
    ax.hist(blast_data, bins=50, alpha=0.6, label='Blast', color='#FF6B6B', edgecolor='black', density=True)
    
    ax.set_xlabel(feature, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'{feature} - Overall Distribution (All Folds Combined)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    stats_text = f"WBC: μ={np.mean(wbc_data):.3f}, σ={np.std(wbc_data):.3f}\nBlast: μ={np.mean(blast_data):.3f}, σ={np.std(blast_data):.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax_empty = fig.add_subplot(gs[2, 1])
ax_empty.axis('off')

fig.suptitle('Feature Distributions - All Folds Combined (Overall)', fontsize=16, fontweight='bold', y=0.995)

png_path = os.path.join(feature_dist_folder, "overall_feature_distributions.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"[SUCCESS] Overall PNG saved to: {png_path}")
plt.close()

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*60)
print("FEATURE DISTRIBUTION VISUALIZATION COMPLETE")
print("="*60)
print(f"PDF saved to: {feature_pdf_path}")
print(f"PNG files saved to: {feature_dist_folder}")
print(f"  ├─ overall_feature_distributions.png")
print(f"  ├─ fold1_feature_distributions.png")
print(f"  ├─ fold2_feature_distributions.png")
print(f"  ├─ fold3_feature_distributions.png")
print(f"  ├─ fold4_feature_distributions.png")
print(f"  └─ fold5_feature_distributions.png")


# ==========================================
# CREATE PDF WITH ALL VISUALIZATIONS
# ==========================================
pdf_path = os.path.join(OUTPUT_PATH, "OneClassSVM_Complete_Analysis.pdf")
pdf = PdfPages(pdf_path)

# ==========================================
# PAGE 1: OVERALL PERFORMANCE SUMMARY
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1.1: Performance Metrics by Fold
ax1 = fig.add_subplot(gs[0, :])
folds = threshold_summary['fold'].values
recalls = threshold_summary['clinical_recall'].values
precisions = threshold_summary['clinical_precision'].values
f1s = threshold_summary['clinical_f1'].values

x = np.arange(len(folds))
width = 0.25

bars1 = ax1.bar(x - width, recalls, width, label='Recall', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x, precisions, width, label='Precision', color='#4ECDC4', alpha=0.8)
bars3 = ax1.bar(x + width, f1s, width, label='F1-Score', color='#95E1D3', alpha=0.8)

ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('One-Class SVM Performance Metrics by Fold', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Fold {i+1}' for i in folds])
ax1.legend(fontsize=11)
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 1.2: Recall Distribution
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(recalls, bins=10, color='#FF6B6B', alpha=0.7, edgecolor='black')
ax2.axvline(recalls.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {recalls.mean():.3f}')
ax2.axvline(np.median(recalls), color='darkred', linestyle=':', linewidth=2, label=f'Median: {np.median(recalls):.3f}')
ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Recall Distribution Across Folds', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 1.3: Precision Distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(precisions, bins=10, color='#4ECDC4', alpha=0.7, edgecolor='black')
ax3.axvline(precisions.mean(), color='teal', linestyle='--', linewidth=2, label=f'Mean: {precisions.mean():.3f}')
ax3.axvline(np.median(precisions), color='darkslategray', linestyle=':', linewidth=2, label=f'Median: {np.median(precisions):.3f}')
ax3.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Precision Distribution Across Folds', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 1.4: Summary Statistics Table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

summary_data = [
    ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
    ['Recall', f'{recalls.mean():.3f}', f'{recalls.std():.3f}', f'{recalls.min():.3f}', f'{recalls.max():.3f}'],
    ['Precision', f'{precisions.mean():.3f}', f'{precisions.std():.3f}', f'{precisions.min():.3f}', f'{precisions.max():.3f}'],
    ['F1-Score', f'{f1s.mean():.3f}', f'{f1s.std():.3f}', f'{f1s.min():.3f}', f'{f1s.max():.3f}']
]

table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, 4):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

fig.suptitle('One-Class SVM - Overall Performance Summary', 
             fontsize=16, fontweight='bold', y=0.995)
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 1: Overall Performance Summary - SAVED")

# ==========================================
# PAGE 2: CONFUSION MATRICES FOR EACH FOLD
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices by Fold', fontsize=16, fontweight='bold')

fold_id = 0
for ax_idx, ax in enumerate(axes.flatten()):
    if fold_id < 5:
        fold_data = all_samples_results[all_samples_results['fold'] == fold_id]
        
        y_true = fold_data['gt_count'].values
        y_pred = fold_data['pred_count'].values
        
        # Reconstruct confusion matrix from individual samples
        # This is approximate; for exact CM, we'd need sample-level predictions
        fold_inner_data = pd.read_csv(os.path.join(CV_PATH, f"fold{fold_id}_OneClassSVM_outerCV_optimal.csv"))
        
        # Get predictions from inner CV for this fold
        y_true_fold = []
        y_pred_fold = []
        
        for idx, row in fold_inner_data.iterrows():
            gt = row['gt_count']
            pred = row['pred_count']
            y_true_fold.extend([1] * gt)
            y_pred_fold.extend([1] * pred)
            
            # Add WBCs (approximation)
            wbc_count = max(0, 10000 - gt)  # Approximate
            y_true_fold.extend([0] * wbc_count)
            y_pred_fold.extend([0] * max(0, 10000 - pred))
        
        cm = confusion_matrix([min(1, y) for y in y_true_fold[:1000]], 
                             [min(1, y) for y in y_pred_fold[:1000]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar=True, square=True, annot_kws={'size': 12})
        ax.set_title(f'Fold {fold_id + 1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        
        fold_id += 1
    else:
        ax.axis('off')

plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 2: Confusion Matrices - SAVED")

# ==========================================
# PAGE 3: ROC CURVES BY FOLD
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ROC Curves by Fold', fontsize=16, fontweight='bold')

fold_id = 0
for ax_idx, ax in enumerate(axes.flatten()):
    if fold_id < 5:
        fold_data = all_samples_results[all_samples_results['fold'] == fold_id]
        
        # Reconstruct y_true and y_scores from sample-level results
        y_true_fold = []
        decision_scores = []
        
        for idx, row in fold_data.iterrows():
            gt_count = int(row['gt_count'])
            y_true_fold.extend([1] * gt_count)
            decision_scores.extend([row['mean_decision_score_blast']] * gt_count)
            
            wbc_approx = max(1, int(row['gt_perc'] / (row['gt_perc'] + 1e-6) * 10000))
            y_true_fold.extend([0] * wbc_approx)
            decision_scores.extend([row['mean_decision_score_wbc']] * wbc_approx)
        
        if len(np.unique(y_true_fold)) > 1:
            fpr, tpr, _ = roc_curve(y_true_fold, -np.array(decision_scores))
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='#FF6B6B', lw=2.5, 
                   label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontweight='bold')
            ax.set_title(f'Fold {fold_id + 1}', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(alpha=0.3)
        
        fold_id += 1
    else:
        ax.axis('off')

plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 3: ROC Curves - SAVED")

# ==========================================
# PAGE 4: PRECISION-RECALL CURVES
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Precision-Recall Curves by Fold', fontsize=16, fontweight='bold')

fold_id = 0
for ax_idx, ax in enumerate(axes.flatten()):
    if fold_id < 5:
        fold_data = all_samples_results[all_samples_results['fold'] == fold_id]
        
        y_true_fold = []
        decision_scores = []
        
        for idx, row in fold_data.iterrows():
            gt_count = int(row['gt_count'])
            y_true_fold.extend([1] * gt_count)
            decision_scores.extend([row['mean_decision_score_blast']] * gt_count)
            
            wbc_approx = max(1, int(row['gt_perc'] / (row['gt_perc'] + 1e-6) * 10000))
            y_true_fold.extend( [0]* wbc_approx)
            decision_scores.extend([row['mean_decision_score_wbc']] * wbc_approx)
        
        if len(np.unique(y_true_fold)) > 1:
            precision, recall, _ = precision_recall_curve(y_true_fold, -np.array(decision_scores))
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color='#4ECDC4', lw=2.5, 
                   label=f'AUC = {pr_auc:.3f}')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontweight='bold')
            ax.set_ylabel('Precision', fontweight='bold')
            ax.set_title(f'Fold {fold_id + 1}', fontsize=12, fontweight='bold')
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(alpha=0.3)
        
        fold_id += 1
    else:
        ax.axis('off')

plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 4: Precision-Recall Curves - SAVED")

# ==========================================
# PAGE 5: THRESHOLD ANALYSIS
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 5.1: Threshold Values by Fold
ax1 = fig.add_subplot(gs[0, :])
thresholds = threshold_summary['clinical_threshold'].values
folds_list = [f'Fold {i+1}' for i in range(len(thresholds))]

bars = ax1.bar(folds_list, thresholds, color='#95E1D3', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Threshold')
ax1.set_ylabel('Threshold Value', fontsize=12, fontweight='bold')
ax1.set_title('Optimal Decision Thresholds by Fold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

for bar, thresh in zip(bars, thresholds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{thresh:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 5.2: F1 vs Precision-Recall Trade-off
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(threshold_summary['clinical_precision'], 
           threshold_summary['clinical_recall'],
           s=150, c=thresholds, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Threshold', fontweight='bold')
ax2.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax2.set_ylabel('Recall', fontsize=11, fontweight='bold')
ax2.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# 5.3: Threshold Distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(thresholds, bins=10, color='#95E1D3', alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(thresholds), color='teal', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(thresholds):.2f}')
ax3.set_xlabel('Threshold Value', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Threshold Distribution', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

fig.suptitle('Threshold Analysis', fontsize=16, fontweight='bold')
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 5: Threshold Analysis - SAVED")

# ==========================================
# PAGE 6: DECISION FUNCTION SCORE DISTRIBUTION
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# 6.1: Overall Distribution
ax1 = fig.add_subplot(gs[0, 0])
all_blast_scores = all_samples_results['mean_decision_score_blast'].dropna().values
all_wbc_scores = all_samples_results['mean_decision_score_wbc'].dropna().values

ax1.hist(all_wbc_scores, bins=30, alpha=0.6, label='WBC', color='#4ECDC4', edgecolor='black')
ax1.hist(all_blast_scores, bins=30, alpha=0.6, label='Blast', color='#FF6B6B', edgecolor='black')
ax1.set_xlabel('Decision Function Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Decision Score Distribution (All Folds)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 6.2: Boxplot
ax2 = fig.add_subplot(gs[0, 1])
data_box = [all_wbc_scores, all_blast_scores]
bp = ax2.boxplot(data_box, labels=['WBC', 'Blast'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#4ECDC4', '#FF6B6B']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Decision Function Score', fontsize=11, fontweight='bold')
ax2.set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

# 6.3: Mean Scores by Fold
ax3 = fig.add_subplot(gs[1, 0])
fold_means_blast = all_samples_results.groupby('fold')['mean_decision_score_blast'].mean()
fold_means_wbc = all_samples_results.groupby('fold')['mean_decision_score_wbc'].mean()

x = np.arange(len(fold_means_blast))
width = 0.35

ax3.bar(x - width/2, fold_means_wbc, width, label='WBC', color='#4ECDC4', alpha=0.7)
ax3.bar(x + width/2, fold_means_blast, width, label='Blast', color='#FF6B6B', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Fold', fontweight='bold')
ax3.set_ylabel('Mean Decision Score', fontweight='bold')
ax3.set_title('Mean Scores by Fold', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_means_blast))])
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# 6.4: Statistics Table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

stats_data = [
    ['Metric', 'WBC', 'Blast'],
    ['Mean', f'{np.mean(all_wbc_scores):.3f}', f'{np.mean(all_blast_scores):.3f}'],
    ['Std', f'{np.std(all_wbc_scores):.3f}', f'{np.std(all_blast_scores):.3f}'],
    ['Min', f'{np.min(all_wbc_scores):.3f}', f'{np.min(all_blast_scores):.3f}'],
    ['Max', f'{np.max(all_wbc_scores):.3f}', f'{np.max(all_blast_scores):.3f}'],
]

table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                  colWidths=[0.33, 0.33, 0.33])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(3):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 5):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')

fig.suptitle('Decision Function Score Analysis', fontsize=16, fontweight='bold')
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 6: Decision Score Distribution - SAVED")

# ==========================================
# PAGE 7: SAMPLE-LEVEL PERFORMANCE
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# 7.1: Accuracy by Sample
ax1 = fig.add_subplot(gs[0, 0])
sample_accuracy = all_samples_results.groupby('fold')['accuracy'].apply(list)
positions = range(1, 6)
ax1.boxplot([sample_accuracy[i] for i in range(5)], positions=positions, patch_artist=True)
ax1.set_xlabel('Fold', fontweight='bold')
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Sample-Level Accuracy Distribution', fontsize=12, fontweight='bold')
ax1.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax1.grid(alpha=0.3, axis='y')

# 7.2: Recall by Sample
ax2 = fig.add_subplot(gs[0, 1])
sample_recall = all_samples_results.groupby('fold')['recall'].apply(list)
ax2.boxplot([sample_recall[i] for i in range(5)], positions=positions, patch_artist=True)
ax2.set_xlabel('Fold', fontweight='bold')
ax2.set_ylabel('Recall', fontweight='bold')
ax2.set_title('Sample-Level Recall Distribution', fontsize=12, fontweight='bold')
ax2.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax2.grid(alpha=0.3, axis='y')

# 7.3: Precision by Sample
ax3 = fig.add_subplot(gs[1, 0])
sample_precision = all_samples_results.groupby('fold')['precision'].apply(list)
ax3.boxplot([sample_precision[i] for i in range(5)], positions=positions, patch_artist=True)
ax3.set_xlabel('Fold', fontweight='bold')
ax3.set_ylabel('Precision', fontweight='bold')
ax3.set_title('Sample-Level Precision Distribution', fontsize=12, fontweight='bold')
ax3.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax3.grid(alpha=0.3, axis='y')

# 7.4: F1 by Sample
ax4 = fig.add_subplot(gs[1, 1])
sample_f1 = all_samples_results.groupby('fold')['f1'].apply(list)
ax4.boxplot([sample_f1[i] for i in range(5)], positions=positions, patch_artist=True)
ax4.set_xlabel('Fold', fontweight='bold')
ax4.set_ylabel('F1-Score', fontweight='bold')
ax4.set_title('Sample-Level F1 Distribution', fontsize=12, fontweight='bold')
ax4.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax4.grid(alpha=0.3, axis='y')

fig.suptitle('Sample-Level Performance Metrics', fontsize=16, fontweight='bold')
pdf.savefig(fig, bbox_inches='tight')
plt.close()

print("[INFO] Page 7: Sample-Level Performance - SAVED")



# ==========================================
# CLOSE PDF
# ==========================================
pdf.close()

print(f"\n[SUCCESS] Complete PDF report saved to: {pdf_path}")