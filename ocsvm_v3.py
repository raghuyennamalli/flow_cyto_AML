import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, ParameterGrid, GroupShuffleSplit
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer,
    precision_recall_curve, roc_curve, roc_auc_score, auc)
from FlowCytometryTools import FCMeasurement
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

# ========================================= SETUP ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/output_svm"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)

features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]

start_time = time.time()
script_start_time = time.time()
# ========================================== HELPER FUNCTIONS ==========================================
def from_fcs(path):
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()
    
def find_optimal_threshold(y_true, y_scores, method='clinical_recall'):
    y_scores_flipped = -y_scores
            
    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores_flipped)
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
    optimal_thresh_raw = -optimal_thresh
    y_pred = (y_scores <= optimal_thresh_raw).astype(int)
    
    metrics = {
        'threshold': optimal_thresh_raw,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)}
    return optimal_thresh, metrics

# ========================================== STEP 0: CREATE AGGREGATED DATASET (2K + 5K) ==========================================
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

else:
    print("[INFO] Pickled data already exists.")
    
# ========================================== STEP 1: LOAD DATA (CORRECTED) ==========================================
pkl_2k = os.path.join(OUTPUT_PATH, "BLAST110_2K.pkl")
data = pd.read_pickle(pkl_2k)


X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]

X_scaled = X.values

print(f"[INFO] Data shape: {X_scaled.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ========================================== # STEP 2: DEFINE CV STRATEGY # ==========================================
gs_80_20 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for cv_idx, test_reserve_idx in gs_80_20.split(X_scaled, y, groups):
    X_cv_pool = X_scaled[cv_idx]              # 880 samples (80%)
    X_test_reserve = X_scaled[test_reserve_idx]  # 220 samples
    y_cv_pool = y.iloc[cv_idx]
    y_test_reserve = y.iloc[test_reserve_idx]
    groups_cv_pool = groups.iloc[cv_idx]
    groups_test_reserve = groups.iloc[test_reserve_idx]
    samples_cv_pool = samples.iloc[cv_idx]

# Second split: Split the 220 into 10% external test + 10% reserve
gs_10_10 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for ext_idx, res_idx in gs_10_10.split(X_test_reserve, y_test_reserve, groups_test_reserve):
    X_external_test = X_test_reserve[ext_idx]     # 110 samples (10%)
    X_reserve = X_test_reserve[res_idx]           # 110 samples (10%)
    y_external_test = y_test_reserve.iloc[ext_idx]
    y_reserve = y_test_reserve.iloc[res_idx]

print(f"[SPLIT] CV Pool: {X_cv_pool.shape[0]} | External: {X_external_test.shape[0]} | Reserve: {X_reserve.shape[0]}")

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)

# ========================================== STEP 3: EXHAUSTIVE HYPERPARAMETER GRID ==========================================
param_grid = {
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", 0.01, 0.1, 0.5], 
    "nu": [0.001, 0.01, 0.03,0.05, 0.10] }
    
print(f"[INFO] Hyperparameter combinations: ~{len(list(ParameterGrid(param_grid)))}")

def process_fold(fold_data):
    """Process one outer fold"""
    fold, train_idx, test_idx = fold_data
    
    X_train = X_cv_pool[train_idx]
    X_test = X_cv_pool[test_idx]
    y_train = y_cv_pool.iloc[train_idx]
    y_test = y_cv_pool.iloc[test_idx]
    train_groups = groups_cv_pool.iloc[train_idx]
    
    X_train_norm = X_train[y_train == 0]
    
    # ===== INNER CV =====
    best_inner_score = -1
    best_inner_params = None
    fold_inner_thresholds = []
    param_list = list(ParameterGrid(param_grid))
    param_scores = {i: [] for i in range(len(param_list))}
    
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(X_train, y_train, train_groups)):
        X_inner_train_full = X_train[inner_train_idx]
        y_inner_train_full = y_train.iloc[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train.iloc[inner_val_idx]
        X_inner_train_norm = X_inner_train_full[y_inner_train_full == 0]
        
        for i, params in enumerate(param_list):
            try:
                model = OneClassSVM(**params)
                model.fit(X_inner_train_norm)
                scores = model.decision_function(X_inner_val)
                if len(np.unique(y_inner_val)) > 1:
                    score = roc_auc_score(y_inner_val, -scores)
                else:
                    score = 0.5
                param_scores[i].append(score)
            except:
                param_scores[i].append(0)
        
        best_fold_score = -1
        best_fold_params = None
        for i, scores_list in param_scores.items():
            if len(scores_list) > 0:
                latest_score = scores_list[-1]
                if latest_score > best_fold_score:
                    best_fold_score = latest_score
                    best_fold_params = param_list[i]
        
        if best_fold_params:
            model_inner = OneClassSVM(**best_fold_params)
            model_inner.fit(X_inner_train_norm)
            scores_inner = model_inner.decision_function(X_inner_val)
            thresh_inner, _ = find_optimal_threshold(y_inner_val.values, scores_inner)
            fold_inner_thresholds.append(thresh_inner)
    
    for i, scores in param_scores.items():
        avg_score = np.mean(scores)
        if avg_score > best_inner_score:
            best_inner_score = avg_score
            best_inner_params = param_list[i]
    if best_inner_params is None:
        best_inner_params = param_list[0] 
        print(f"[WARNING] Fold {fold}: No valid params found, using first combo")
    
    # ===== OUTER TEST =====
    best_svm = OneClassSVM(**best_inner_params)
    best_svm.fit(X_train_norm)
    anomaly_scores_test = best_svm.decision_function(X_test)
    
    fold_threshold = np.median(fold_inner_thresholds) if fold_inner_thresholds else 0.0
    preds = (anomaly_scores_test < fold_threshold).astype(int)
    
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    # ===== CONFUSION MATRIX FOR THIS OUTER FOLD =====
    y_true = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(preds).reshape(-1)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== Confusion Matrix: Fold {fold} ===")
    print(cm)

    print(classification_report(
        y_true,
        y_pred,
        target_names=["Non-blast", "Blast"],
        zero_division=0 ))
    
    # ===== PER-SAMPLE =====
    sample_ids = samples_cv_pool.iloc[test_idx].unique()
    model_results = []
    for sid in sample_ids:
        subset = data[data["sample_id"] == sid]
        X_sub = subset[features].values
        y_sub = subset["Blast"].values
        scores_sub = best_svm.decision_function(X_sub)
        pred_sub = (scores_sub < fold_threshold).astype(int)
        
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
            "mean_decision_score_nonblast": np.mean(scores_sub[y_sub == 0]) if np.any(y_sub == 0) else np.nan,
            "optimal_threshold": fold_threshold}
        model_results.append(result_dict)
    
    return {'fold_summary': {
            'fold': fold,
            'best_params': str(best_inner_params),
            'inner_cv_threshold': fold_threshold,
            'precision': prec,
            'recall': rec,
            'f1': f1},
        'thresholds': fold_inner_thresholds,
        'samples': model_results,
        'best_params': best_inner_params}

# ========================================== STEP 4: PARALLEL CV LOOP ==========================================
print(f"\n[PARALLEL] Preparing 5 outer folds for parallelization...")

# Prepare fold data
fold_data_list = [
    (fold, train_idx, test_idx) 
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_cv_pool, y_cv_pool, groups_cv_pool))]

# Run in parallel
print(f"[PARALLEL] Running on all available cores...")
results = Parallel(n_jobs=-1, verbose=10, backend='loky')(
    delayed(process_fold)(fold_data) 
    for fold_data in fold_data_list)

print(f"\n[SUCCESS] All 5 folds completed")

# Aggregate results
threshold_results = []
all_fold_results = []
all_inner_cv_thresholds = []
all_best_params = []

for result in results:
    threshold_results.append(result['fold_summary'])
    all_fold_results.extend(result['samples'])
    all_inner_cv_thresholds.extend(result['thresholds'])
    all_best_params.append(result['best_params']) 

fold_summary_df = pd.DataFrame(threshold_results)
best_fold_idx = fold_summary_df['f1'].idxmax()
best_inner_params = all_best_params[best_fold_idx]

# Save results (same as before)
pd.DataFrame(threshold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "nested_cv_unbiased_results.csv"), index=False)

pd.DataFrame(all_fold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "cv_pool_all_samples_results.csv"), index=False)
all_samples_results = pd.DataFrame(all_fold_results)

print(f"\n[INFO] Results saved: {len(threshold_results)} fold summaries, {len(all_fold_results)} sample results")

# ========================================== STEP 5: FINALIZE THRESHOLD ==========================================
# ✅ NEW: Aggregate threshold across ALL inner CV folds

final_threshold = np.median(all_inner_cv_thresholds)

print(f"\n[THRESHOLD FINALIZATION]")
print(f"  All inner CV thresholds: {all_inner_cv_thresholds}")
print(f"  Final threshold (median): {final_threshold:.6f}")
print(f"  Threshold range: [{np.min(all_inner_cv_thresholds):.6f}, {np.max(all_inner_cv_thresholds):.6f}]")
print(f"  Threshold std: {np.std(all_inner_cv_thresholds):.6f}")

# ========================================== STEP 6: REFIT ON FULL CV POOL ==========================================
# ✅ NEW: Train final production model using all CV pool data

print(f"\n[FINAL MODEL REFIT]")
print(f"Refitting model on entire CV pool ({X_cv_pool.shape[0]} samples)...")

# Train on ALL normals in CV pool
X_cv_pool_norm = X_cv_pool[y_cv_pool == 0]

  
final_model = OneClassSVM(**best_inner_params)
final_model.fit(X_cv_pool_norm)

print(f"  Final model support vectors: {len(final_model.support_vectors_)}")
print(f"  % of training data: {len(final_model.support_vectors_) / len(X_cv_pool_norm) * 100:.2f}%")

# ========================================== STEP 7: EXTERNAL TEST EVALUATION ==========================================
# ✅ CRITICAL: Final unbiased evaluation on external test (10%)
# NEVER modify threshold on this data

print(f"\n{'='*60}")
print("EXTERNAL TEST EVALUATION (UNBIASED)")
print(f"{'='*60}")

# Predict on external test using final threshold
anomaly_scores_external = final_model.decision_function(X_external_test)
preds_external = (anomaly_scores_external < final_threshold).astype(int)

# Compute UNBIASED metrics
precision_external = precision_score(y_external_test, preds_external, zero_division=0)
recall_external = recall_score(y_external_test, preds_external, zero_division=0)
f1_external = f1_score(y_external_test, preds_external, zero_division=0)
accuracy_external = accuracy_score(y_external_test, preds_external)

print(f"\nExternal Test Performance (UNBIASED):")
print(f"  Precision: {precision_external:.3f}")
print(f"  Recall: {recall_external:.3f}")
print(f"  F1-Score: {f1_external:.3f}")
print(f"  Accuracy: {accuracy_external:.3f}")
print(f"\nConfusion Matrix (External Test):")
cm_external = confusion_matrix(y_external_test, preds_external)
print(cm_external)
print(f"\nClassification Report (External Test):")
print(classification_report(y_external_test, preds_external, 
                          target_names=["Non-blast", "Blast"]))

# Score distribution on external test
nonblast_scores_ext = anomaly_scores_external[y_external_test.values == 0]
blast_scores_ext = anomaly_scores_external[y_external_test.values == 1]

print(f"\nDecision Function Scores (External Test):")
print(f"Non-blast: mean={np.mean(nonblast_scores_ext):.6f}, std={np.std(nonblast_scores_ext):.6f}")
print(f"  % > 0: {np.sum(nonblast_scores_ext > 0) / len(nonblast_scores_ext) * 100:.2f}%")
print(f"Blast: mean={np.mean(blast_scores_ext):.6f}, std={np.std(blast_scores_ext):.6f}")
print(f"  % < 0: {np.sum(blast_scores_ext < 0) / len(blast_scores_ext) * 100:.2f}%")

# Save external test results
external_test_results = {
    'phase': 'External Test',
    'threshold': final_threshold,
    'precision': precision_external,
    'recall': recall_external,
    'f1': f1_external,
    'accuracy': accuracy_external,
    'sample_count': len(y_external_test),
    'true_negatives': cm_external[0, 0],
    'false_positives': cm_external[0, 1],
    'false_negatives': cm_external[1, 0],
    'true_positives': cm_external[1, 1]}

pd.DataFrame([external_test_results]).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "external_test_results.csv"), index=False)

print(f"\n[SUCCESS] External test results saved")

# ========================================== SAVE FINAL SUMMARIES ==========================================
# ========================================== STEP 8: SAVE SUMMARIES ==========================================

# Save CV results
pd.DataFrame(threshold_results).to_csv(
    os.path.join(OUTPUT_PATH, "CV", "nested_cv_unbiased_results.csv"), index=False)


# Summary statistics
avg_cv_results = pd.DataFrame(threshold_results)
print(f"\n{'='*60}")
print("NESTED CV RESULTS (CV POOL - 80%)")
print(f"{'='*60}")
print(f"Average F1: {avg_cv_results['f1'].mean():.3f} ± {avg_cv_results['f1'].std():.3f}")
print(f"Average Precision: {avg_cv_results['precision'].mean():.3f} ± {avg_cv_results['precision'].std():.3f}")
print(f"Average Recall: {avg_cv_results['recall'].mean():.3f} ± {avg_cv_results['recall'].std():.3f}")

print(f"\n{'='*60}")
print("COMPARISON: CV POOL vs EXTERNAL TEST")
print(f"{'='*60}")
print(f"CV Pool F1:         {avg_cv_results['f1'].mean():.3f} ± {avg_cv_results['f1'].std():.3f}")
print(f"External Test F1:   {f1_external:.3f}")
print(f"Difference:         {(avg_cv_results['f1'].mean() - f1_external):.3f}")
print(f"\nNote: External test represents true generalization performance")
print(f"      on completely independent data (never seen during model development)")


# ========================================== FEATURE DISTRIBUTION PLOTS - CONSOLIDATED ==========================================
sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (14, 10), 'font.size': 10})

print("\n" + "="*60 + "GENERATING FEATURE DISTRIBUTION PLOTS" + "="*60)
feature_dist_folder = os.path.join(OUTPUT_PATH, "feature_distributions")
os.makedirs(feature_dist_folder, exist_ok=True)

# ========================================== HELPER FUNCTION ==========================================
def create_feature_plot(fold_id=None, save_pdf=False, save_png=False, pdf_writer=None):
    """Generate feature distribution plot for a specific fold or overall."""
    is_overall = (fold_id is None)
    if is_overall:        # Overall: Use all data
        X_data = {feat: data[feat].values for feat in features}
        y_data = data["Blast"].values
        title_suffix = "All Folds Combined (Overall)"
        filename = "overall_feature_distributions.png"
    else:               # Fold-specific: Get train/test split
        outer_cv_splits = list(outer_cv.split(X_cv_pool, y_cv_pool, groups_cv_pool))
        train_idx, test_idx = outer_cv_splits[fold_id]
        
        X_train_fold =  X_cv_pool[train_idx]
        X_test_fold = X_cv_pool[test_idx]
        y_train_fold = y_cv_pool.iloc[train_idx]
        y_test_fold = y_cv_pool.iloc[test_idx]
        
        title_suffix = f"Fold {fold_id + 1}: Feature Distributions - Train vs Test"
        filename = f"fold{fold_id + 1}_feature_distributions.png"
    fig = plt.figure(figsize=(16, 12))
    num_features = len(features)
    num_rows = (num_features + 1) // 2  # Ceiling division
    num_cols = 2
    gs = GridSpec(num_rows, num_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    for feat_idx, feature in enumerate(features):
        row_idx = feat_idx // num_cols
        col_idx = feat_idx % num_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        if is_overall:
            # Plot Non-blast vs Blast
            nonblast_data = X_data[feature][y_data == 0]
            blast_data = X_data[feature][y_data == 1]
            
            ax.hist(nonblast_data, bins=50, alpha=0.6, label='Non-blast', color='#4ECDC4', edgecolor='black', density=True)
            ax.hist(blast_data, bins=50, alpha=0.6, label='Blast',   color='#FF6B6B', edgecolor='black', density=True)
            stats_text = (f"Non-blast: μ={np.mean(nonblast_data):.3f}, σ={np.std(nonblast_data):.3f}\n" f"Blast: μ={np.mean(blast_data):.3f}, σ={np.std(blast_data):.3f}")
        else:
            # Plot Train vs Test for Non-blast and Blast
            X_train_feat = X_train_fold[:, feat_idx]
            X_test_feat = X_test_fold[:, feat_idx]
            
            train_nonblast = X_train_feat[y_train_fold == 0]
            train_blast = X_train_feat[y_train_fold == 1]
            test_nonblast = X_test_feat[y_test_fold == 0]
            test_blast = X_test_feat[y_test_fold == 1]
            
            ax.hist(train_nonblast, bins=40, alpha=0.5, label='Train Non-blast',  color='#4ECDC4', edgecolor='black', density=True)
            ax.hist(test_nonblast, bins=40, alpha=0.5, label='Test Non-blast',  color='#2E9B96', edgecolor='black', density=True, histtype='step', linewidth=2)
            ax.hist(train_blast, bins=40, alpha=0.5, label='Train Blast',   color='#FF6B6B', edgecolor='black', density=True)
            ax.hist(test_blast, bins=40, alpha=0.5, label='Test Blast',  color='#CC5555', edgecolor='black', density=True, histtype='step', linewidth=2)
            
            stats_text = (f"Train Non-blast: μ={np.mean(train_nonblast):.3f}, σ={np.std(train_nonblast):.3f}\n"
                         f"Test Non-blast: μ={np.mean(test_nonblast):.3f}, σ={np.std(test_nonblast):.3f}\n"
                         f"Train Blast: μ={np.mean(train_blast):.3f}, σ={np.std(train_blast):.3f}\n"
                         f"Test Blast: μ={np.mean(test_blast):.3f}, σ={np.std(test_blast):.3f}")
        # Common formatting
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{feature} - {title_suffix}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10 if not is_overall else 11, loc='upper right')
        ax.grid(alpha=0.3, axis='y')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8 if not is_overall else 9,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)) 
    # Empty subplot for 5th position
    if num_features % 2 == 1:  # Only if odd number of features
      fig.add_subplot(gs[num_rows - 1, num_cols - 1]).axis('off')
    fig.suptitle(title_suffix, fontsize=16, fontweight='bold', y=0.995)
    # Save outputs
    if save_pdf and pdf_writer:
        pdf_writer.savefig(fig, bbox_inches='tight')
    if save_png:
        png_path = os.path.join(feature_dist_folder, filename)
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"[SUCCESS] {filename} saved")
    plt.close()

# ========================================== GENERATE PDF WITH ALL FEATURE PLOTS ==========================================
feature_pdf_path = os.path.join(OUTPUT_PATH, "Feature_Distributions_AllFolds.pdf")
with PdfPages(feature_pdf_path) as feature_pdf: 
    # Page 1: Overall distribution
    print("[INFO] Generating overall feature distributions...")
    create_feature_plot(fold_id=None, save_pdf=True, save_png=True, pdf_writer=feature_pdf)
    print("[INFO] Page 1: Overall Feature Distributions - SAVED")
    # Pages 2-6: Individual folds
    for fold_id in range(5):
        print(f"[INFO] Generating feature distributions for Fold {fold_id + 1}...")
        create_feature_plot(fold_id=fold_id, save_pdf=True, save_png=True, pdf_writer=feature_pdf)
        print(f"[INFO] Fold {fold_id + 1}: Feature Distributions - SAVED")
    
    # Page 7: Statistics summary table
    print("[INFO] Generating feature statistics summary...")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    summary_stats = [[feature,
        f"{np.mean(data[feature].values[data['Blast'] == 0]):.4f}",
        f"{np.std(data[feature].values[data['Blast'] == 0]):.4f}",
        f"{np.min(data[feature].values[data['Blast'] == 0]):.4f}",
        f"{np.max(data[feature].values[data['Blast'] == 0]):.4f}",
        f"{np.mean(data[feature].values[data['Blast'] == 1]):.4f}",
        f"{np.std(data[feature].values[data['Blast'] == 1]):.4f}",
        f"{np.min(data[feature].values[data['Blast'] == 1]):.4f}",
        f"{np.max(data[feature].values[data['Blast'] == 1]):.4f}"]
        for feature in features]
    
    table_data = [['Feature', 'Non-blast Mean', 'Non-blast Std', 'Non-blast Min', 'Non-blast Max', 
                   'Blast Mean', 'Blast Std', 'Blast Min', 'Blast Max']] + summary_stats
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',colWidths=[0.12] + [0.11]*8)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    # Style header and alternate rows
    for i in range(9):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, 6):
        for j in range(9):
            table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else '#FFFFFF')
    fig.suptitle('Feature Statistics Summary - All Folds', fontsize=16, fontweight='bold')
    feature_pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("[INFO] Page 7: Feature Statistics Summary - SAVED")

print(f"\n[SUCCESS] Feature distributions PDF saved to: {feature_pdf_path}")
print(f"PNG files saved to: {feature_dist_folder}")

# ========================================== CREATE PDF WITH ALL VISUALIZATIONS ==========================================
pdf_path = os.path.join(OUTPUT_PATH, "OneClassSVM_Complete_Analysis.pdf")
pdf = PdfPages(pdf_path)
# ========================================== PAGE 1: OVERALL PERFORMANCE SUMMARY ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :]) # 1.1: Performance Metrics by Fold
threshold_summary = pd.DataFrame(threshold_results)

folds = threshold_summary['fold'].values
recalls = threshold_summary['recall'].values
precisions = threshold_summary['precision'].values
f1s = threshold_summary['f1'].values

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

for bars in [bars1, bars2, bars3]:          # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,f'{height:.2f}', ha='center', va='bottom', fontsize=9)
ax2 = fig.add_subplot(gs[1, 0])     # 1.2: Recall Distribution
ax2.hist(recalls, bins=10, color='#FF6B6B', alpha=0.7, edgecolor='black')
ax2.axvline(recalls.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {recalls.mean():.3f}')
ax2.axvline(np.median(recalls), color='darkred', linestyle=':', linewidth=2, label=f'Median: {np.median(recalls):.3f}')
ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Recall Distribution Across Folds', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])     # 1.3: Precision Distribution
ax3.hist(precisions, bins=10, color='#4ECDC4', alpha=0.7, edgecolor='black')
ax3.axvline(precisions.mean(), color='teal', linestyle='--', linewidth=2, label=f'Mean: {precisions.mean():.3f}')
ax3.axvline(np.median(precisions), color='darkslategray', linestyle=':', linewidth=2, label=f'Median: {np.median(precisions):.3f}')
ax3.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Precision Distribution Across Folds', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax4 = fig.add_subplot(gs[2, :])     # 1.4: Summary Statistics Table
ax4.axis('off')

summary_data = [['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
    ['Recall', f'{recalls.mean():.3f}', f'{recalls.std():.3f}', f'{recalls.min():.3f}', f'{recalls.max():.3f}'],
    ['Precision', f'{precisions.mean():.3f}', f'{precisions.std():.3f}', f'{precisions.min():.3f}', f'{precisions.max():.3f}'],
    ['F1-Score', f'{f1s.mean():.3f}', f'{f1s.std():.3f}', f'{f1s.min():.3f}', f'{f1s.max():.3f}']]
table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

for i in range(5):  
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, 4):       # Alternate row colors
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
fig.suptitle('One-Class SVM - Overall Performance Summary', fontsize=16, fontweight='bold', y=0.995)
pdf.savefig(fig, bbox_inches='tight')
plt.close()
print("[INFO] Page 1: Overall Performance Summary - SAVED")

# ========================================== PAGE 2: CONFUSION MATRICES FOR EACH FOLD ==========================================
'''fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
            
            # Add Non-blasts (approximation)
            nonblast_count = max(0, 10000 - gt)  # Approximate
            y_true_fold.extend([0] * nonblast_count)
            y_pred_fold.extend([0] * max(0, 10000 - pred))
        
        cm = confusion_matrix([min(1, y) for y in y_true_fold[:1000]], [min(1, y) for y in y_pred_fold[:1000]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, square=True, annot_kws={'size': 12})
        ax.set_title(f'Fold {fold_id + 1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        
        fold_id += 1
    else:
        ax.axis('off')
plt.tight_layout()
pdf.savefig(fig, bbox_inches='tight')
plt.close()
print("[INFO] Page 2: Confusion Matrices - SAVED")'''

# ========================================== PAGE 3: ROC CURVES BY FOLD (CORRECTED) ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ROC Curves by Fold', fontsize=16, fontweight='bold')

fold_id = 0
for ax_idx, ax in enumerate(axes.flatten()):
    if fold_id < 5:
        fold_data = all_samples_results[all_samples_results['fold'] == fold_id]
        
        # Reconstruct ACTUAL event-level data for this fold
        y_true_fold = []
        decision_scores = []
        
        for idx, row in fold_data.iterrows():
            sample_id = row['sample_id']
            
            # Get the ACTUAL events from your original data
            sample_subset = data[data['sample_id'] == sample_id]
            X_sample = sample_subset[features].values
            y_sample = sample_subset['Blast'].values
            
            # Get REAL decision scores (not aggregated means)
            scores_sample = final_model.decision_function(X_sample)
            
            # Extend with actual individual events
            y_true_fold.extend(y_sample)
            decision_scores.extend(scores_sample)
        
        if len(np.unique(y_true_fold)) > 1:
            # ROC curve: higher decision scores = more anomalous (blast-like)
            # So we use decision_scores directly (not negated)
            fpr, tpr, _ = roc_curve(y_true_fold, -np.array(decision_scores)) 
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='#FF6B6B', lw=2.5, label=f'AUC = {roc_auc:.3f}')
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
print("[INFO] Page 3: ROC Curves (CORRECTED) - SAVED")


# ========================================== PAGE 4: PRECISION-RECALL CURVES ==========================================
# ========================================== PAGE 4: PRECISION-RECALL CURVES (CORRECTED) ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Precision-Recall Curves by Fold', fontsize=16, fontweight='bold')

fold_id = 0
for ax_idx, ax in enumerate(axes.flatten()):
    if fold_id < 5:
        fold_data = all_samples_results[all_samples_results['fold'] == fold_id]
        
        # Get the actual events from your original data for this fold
        y_true_fold = []
        decision_scores_fold = []
        
        for idx, row in fold_data.iterrows():
            sample_id = row['sample_id']
            
            # Get ALL events for this sample from original data
            sample_subset = data[data['sample_id'] == sample_id]
            X_sample = sample_subset[features].values
            y_sample = sample_subset['Blast'].values
            
            # Get decision scores from the fold's best model
            # (You need to save fold-specific models or refit here)
            # For now, use the mean scores as proxies
            scores_sample = final_model.decision_function(X_sample)
            
            # Add ACTUAL individual events (not aggregated)
            y_true_fold.extend(y_sample)
            decision_scores_fold.extend(scores_sample)
        
        if len(np.unique(y_true_fold)) > 1:
            precision, recall, _ = precision_recall_curve(y_true_fold, -np.array(decision_scores_fold))
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color='#4ECDC4', lw=2.5, label=f'AUC = {pr_auc:.3f}')
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
print("[INFO] Page 4: Precision-Recall Curves (CORRECTED) - SAVED")


# ========================================== PAGE 5: THRESHOLD ANALYSIS ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 5.1: Threshold Values by Fold
ax1 = fig.add_subplot(gs[0, :])
thresholds = threshold_summary['inner_cv_threshold'].values
folds_list = [f'Fold {i+1}' for i in range(len(thresholds))]

bars = ax1.bar(folds_list, thresholds, color='#95E1D3', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Threshold')
ax1.set_ylabel('Threshold Value', fontsize=12, fontweight='bold')
ax1.set_title('Optimal Decision Thresholds by Fold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

for bar, thresh in zip(bars, thresholds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{thresh:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 5.2: F1 vs Precision-Recall Trade-off
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(threshold_summary['precision'], threshold_summary['recall'],s=150, c=thresholds, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Threshold', fontweight='bold')
ax2.set_xlabel('Precision', fontsize=11, fontweight='bold')
ax2.set_ylabel('Recall', fontsize=11, fontweight='bold')
ax2.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# 5.3: Threshold Distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(thresholds, bins=10, color='#95E1D3', alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(thresholds), color='teal', linestyle='--', linewidth=2, label=f'Mean: {np.mean(thresholds):.2f}')
ax3.set_xlabel('Threshold Value', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Threshold Distribution', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

fig.suptitle('Threshold Analysis', fontsize=16, fontweight='bold')
pdf.savefig(fig, bbox_inches='tight')
plt.close()
print("[INFO] Page 5: Threshold Analysis - SAVED")

# ========================================== PAGE 6: DECISION FUNCTION SCORE DISTRIBUTION ==========================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# 6.1: Overall Distribution
ax1 = fig.add_subplot(gs[0, 0])
all_blast_scores = all_samples_results['mean_decision_score_blast'].dropna().values
all_nonblast_scores = all_samples_results['mean_decision_score_nonblast'].dropna().values

ax1.hist(all_nonblast_scores, bins=30, alpha=0.6, label='Non-blast', color='#4ECDC4', edgecolor='black')
ax1.hist(all_blast_scores, bins=30, alpha=0.6, label='Blast', color='#FF6B6B', edgecolor='black')
ax1.set_xlabel('Decision Function Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Decision Score Distribution (All Folds)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 6.2: Boxplot
ax2 = fig.add_subplot(gs[0, 1])
data_box = [all_nonblast_scores, all_blast_scores]
bp = ax2.boxplot(data_box, labels=['Non-blast', 'Blast'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#4ECDC4', '#FF6B6B']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Decision Function Score', fontsize=11, fontweight='bold')
ax2.set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

# 6.3: Mean Scores by Fold
ax3 = fig.add_subplot(gs[1, 0])
fold_means_blast = all_samples_results.groupby('fold')['mean_decision_score_blast'].mean()
fold_means_nonblast = all_samples_results.groupby('fold')['mean_decision_score_nonblast'].mean()

x = np.arange(len(fold_means_blast))
width = 0.35

ax3.bar(x - width/2, fold_means_nonblast, width, label='Non-blast', color='#4ECDC4', alpha=0.7)
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
    ['Metric', 'Non-blast', 'Blast'],
    ['Mean', f'{np.mean(all_nonblast_scores):.3f}', f'{np.mean(all_blast_scores):.3f}'],
    ['Std', f'{np.std(all_nonblast_scores):.3f}', f'{np.std(all_blast_scores):.3f}'],
    ['Min', f'{np.min(all_nonblast_scores):.3f}', f'{np.min(all_blast_scores):.3f}'],
    ['Max', f'{np.max(all_nonblast_scores):.3f}', f'{np.max(all_blast_scores):.3f}']]

table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',colWidths=[0.33, 0.33, 0.33])
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

# ========================================== PAGE 7: SAMPLE-LEVEL PERFORMANCE ==========================================
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
# ========================================== CLOSE PDF ==========================================
pdf.close()
print(f"\n[SUCCESS] Complete PDF report saved to: {pdf_path}")


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