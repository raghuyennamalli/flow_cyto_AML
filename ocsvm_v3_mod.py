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
import pickle

# ========================================= SETUP ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/Changed_para/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "SVM_CV"), exist_ok=True)

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
    X_external_test = X_scaled[test_reserve_idx]  # 220 samples
    y_cv_pool = y.iloc[cv_idx]
    y_external_test = y.iloc[test_reserve_idx]
    groups_cv_pool = groups.iloc[cv_idx]
    groups_external = groups.iloc[test_reserve_idx]
    samples_cv_pool = samples.iloc[cv_idx]
    samples_external= samples.iloc[test_reserve_idx]

print(f"[SPLIT] CV Pool: {X_cv_pool.shape[0]} | External: {X_external_test.shape[0]}")

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)

# ========================================== STEP 3: EXHAUSTIVE HYPERPARAMETER GRID ==========================================
param_grid = {
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", 0.01, 0.1, 0.5], 
    "nu": [0.001, 0.01, 0.03,0.05, 0.10] }
    
print(f"[INFO] Hyperparameter combinations: ~{len(list(ParameterGrid(param_grid)))}")

def process_fold(fold_data):
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
    acc= accuracy_score(y_test, preds)
    
    # ===== CONFUSION MATRIX FOR THIS OUTER FOLD =====
    y_true = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(preds).reshape(-1)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== Confusion Matrix: Fold {fold} ===")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["Non-blast", "Blast"], zero_division=0 ))
  
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
            "gt_count": int(np.sum(y_sub == 1)), "gt_perc": float(np.mean(y_sub == 1)),
            "pred_count": int(np.sum(pred_sub == 1)),
            "pred_perc": float(np.mean(pred_sub == 1)), "accuracy": accuracy_score(y_sub, pred_sub),
            "precision": precision_score(y_sub, pred_sub, zero_division=0),
            "recall": recall_score(y_sub, pred_sub, zero_division=0),
            "f1": f1_score(y_sub, pred_sub, zero_division=0),
            "mean_decision_score_blast": np.mean(scores_sub[y_sub == 1]) if np.any(y_sub == 1) else np.nan,
            "mean_decision_score_nonblast": np.mean(scores_sub[y_sub == 0]) if np.any(y_sub == 0) else np.nan,
            "optimal_threshold": fold_threshold}
        model_results.append(result_dict)
    
    return {'fold_summary': {'fold': fold,
            'best_params': str(best_inner_params),
            'inner_cv_threshold': fold_threshold,
            'precision': prec, 'recall': rec,'f1': f1, 'accuracy' : acc},
        'thresholds': fold_inner_thresholds, 'samples': model_results,
        'best_params': best_inner_params}

# ========================================== STEP 4: PARALLEL CV LOOP ==========================================
print(f"\n[PARALLEL] Preparing 5 outer folds for parallelization...")
fold_data_list = [(fold, train_idx, test_idx) 
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_cv_pool, y_cv_pool, groups_cv_pool))]

print(f"[PARALLEL] Running on all available cores...")
results = Parallel(n_jobs=-1, verbose=10, backend='loky')(
    delayed(process_fold)(fold_data) for fold_data in fold_data_list)
print(f"\n[SUCCESS] All 5 folds completed")

fold_summaries, all_samples, inner_cv_thresholds, best_params_list = [],[],[], []
for result in results:
    fold_summaries.append(result['fold_summary'])
    all_samples.extend(result['samples'])
    inner_cv_thresholds.extend(result['thresholds'])
    best_params_list.append(result['best_params']) 

fold_df = pd.DataFrame(fold_summaries)
all_samples_results = pd.DataFrame(all_samples)
best_fold_idx = fold_df['f1'].idxmax()
best_inner_params = best_params_list[best_fold_idx]

fold_df.to_csv(os.path.join(OUTPUT_PATH, "SVM_CV", "nested_cv_unbiased_results.csv"), index=False)
all_samples_results.to_csv(os.path.join(OUTPUT_PATH, "SVM_CV", "cv_pool_all_samples_results.csv"), index=False)

print(f"\n[INFO] Results saved: {len(fold_df)} fold summaries, {len(all_samples_results)} sample results")

# ========================================== STEP 5: FINALIZE THRESHOLD ==========================================
final_threshold = np.median(inner_cv_thresholds)
print(f"\n[THRESHOLD FINALIZATION]")
print(f"  Final threshold (median): {final_threshold:.6f}")
print(f"  Threshold range: [{np.min(inner_cv_thresholds):.6f}, {np.max(inner_cv_thresholds):.6f}]")
print(f"  Threshold std: {np.std(inner_cv_thresholds):.6f}")

# ========================================== STEP 6: REFIT ON FULL CV POOL ==========================================
print(f"\n[FINAL MODEL REFIT]")
X_cv_pool_norm = X_cv_pool[y_cv_pool == 0]
final_model = OneClassSVM(**best_inner_params)
final_model.fit(X_cv_pool_norm)

model_bundle = {
    "model": final_model,
    "features": features,
    "threshold": final_threshold,
    "model_type": "OneClassSVM",
    "params": best_inner_params}

model_path = os.path.join(OUTPUT_PATH, "oneclass_svm_final.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"  Final model support vectors: {len(final_model.support_vectors_)}")
print(f"  % of training data: {len(final_model.support_vectors_) / len(X_cv_pool_norm) * 100:.2f}%")

# ========================================== STEP 7: EXTERNAL TEST EVALUATION ==========================================
print("\n" + "="*20 + "EXTERNAL TEST EVALUATION (UNBIASED)" + '='*20)

anomaly_scores_external = final_model.decision_function(X_external_test)
preds_external = (anomaly_scores_external < final_threshold).astype(int)
cm_external = confusion_matrix(y_external_test, preds_external)
metrics_ext = {'precision': precision_score(y_external_test, preds_external, zero_division=0),
    'recall': recall_score(y_external_test, preds_external, zero_division=0),
    'f1': f1_score(y_external_test, preds_external, zero_division=0),
    'accuracy': accuracy_score(y_external_test, preds_external)}
blast_prevalence = (y_external_test == 1).mean() * 100
print(f"% blast prevalence: {blast_prevalence:.2f}%")
print(f"  Precision: {metrics_ext['precision']:.3f}")
print(f"  Recall: {metrics_ext['recall']:.3f}")
print(f"  F1-Score: {metrics_ext['f1']:.3f}")
print(f"  Accuracy: {metrics_ext['accuracy']:.3f}")
print(f"\nConfusion Matrix (External Test):")
print(cm_external)
print(f"\nClassification Report (External Test):")
print(classification_report(y_external_test, preds_external, target_names=["Non-blast", "Blast"]))


if len(np.unique(y_external_test)) > 1:
    fpr_ext, tpr_ext, _ = roc_curve(y_external_test, -np.array(anomaly_scores_external))
    metrics_ext['roc_auc'] = auc(fpr_ext, tpr_ext)
    precision_curve_ext, recall_curve_ext, _ = precision_recall_curve(y_external_test, -np.array(anomaly_scores_external))
    metrics_ext['pr_auc'] = auc(recall_curve_ext, precision_curve_ext)
else:
    fpr_ext, tpr_ext = [0, 1], [0, 1]
    precision_curve_ext, recall_curve_ext = [1, 0], [0, 1]
    metrics_ext['roc_auc'] = metrics_ext['pr_auc'] = 0
    
# Score distribution on external test
nonblast_scores_ext = anomaly_scores_external[y_external_test.values == 0]
blast_scores_ext = anomaly_scores_external[y_external_test.values == 1]
print(f"\nDecision Function Scores (External Test):")
print(f"Non-blast: mean={np.mean(nonblast_scores_ext):.6f}, std={np.std(nonblast_scores_ext):.6f}")
print(f"  % > 0: {np.sum(nonblast_scores_ext > 0) / len(nonblast_scores_ext) * 100:.2f}%")
print(f"Blast: mean={np.mean(blast_scores_ext):.6f}, std={np.std(blast_scores_ext):.6f}")
print(f"  % < 0: {np.sum(blast_scores_ext < 0) / len(blast_scores_ext) * 100:.2f}%")

external_test_results = {
    'phase': 'External Test',
    'accuracy': metrics_ext['accuracy'],
    'threshold': final_threshold,
    'precision': metrics_ext['precision'],
    'recall': metrics_ext['recall'],
    'f1': metrics_ext['f1'],
    'sample_count': len(y_external_test),
    'true_negatives': cm_external[0, 0],
    'false_positives': cm_external[0, 1],
    'false_negatives': cm_external[1, 0],
    'true_positives': cm_external[1, 1]}

pd.DataFrame([external_test_results]).to_csv(os.path.join(OUTPUT_PATH, "SVM_CV", "external_test_results.csv"), index=False)
print(f"\n[SUCCESS] External test results saved")

# ========================================== STEP 8: SAVE SUMMARIES ==========================================
print(f"\n" + "="*20 +"NESTED CV RESULTS (CV POOL - 80%)" + "="*20)
print(f"Average Accuracy: {fold_df['accuracy'].mean():.3f} ± {fold_df['accuracy'].std():.3f}")
print(f"Average F1: {fold_df['f1'].mean():.3f} ± {fold_df['f1'].std():.3f}")
print(f"Average Precision: {fold_df['precision'].mean():.3f} ± {fold_df['precision'].std():.3f}")
print(f"Average Recall: {fold_df['recall'].mean():.3f} ± {fold_df['recall'].std():.3f}")
print(f"External Test F1:   {metrics_ext['f1']:.3f}")
print(f"Difference:         {(fold_df['f1'].mean() - metrics_ext['f1']):.3f}")

#============================ FEATURE DISTRIBUTIONS - OPTIMIZED ==========================================
feature_dist_folder = os.path.join(OUTPUT_PATH, "feature_distributions")
os.makedirs(feature_dist_folder, exist_ok=True)
def plot_features(fold_id=None):
    is_overall = fold_id is None
    if is_overall:
        X_plot = X_cv_pool
        y_plot = y_cv_pool.values
        title = "All Folds Combined"
    else:
        train_idx, test_idx = list(outer_cv.split(X_cv_pool, y_cv_pool, groups_cv_pool))[fold_id]
        X_plot = np.vstack([X_cv_pool[train_idx], X_cv_pool[test_idx]])
        y_plot = np.hstack([y_cv_pool.iloc[train_idx].values, y_cv_pool.iloc[test_idx].values])
        title = f"Fold {fold_id + 1}"   
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()   
    for feat_idx, feature in enumerate(features):
        ax = axes[feat_idx]
        for label, color, name in [(0, '#4ECDC4', 'Non-blast'), (1, '#FF6B6B', 'Blast')]:
            data = X_plot[:, feat_idx][y_plot == label]
            ax.hist(data, bins=40, alpha=0.6, label=name, color=color, edgecolor='black', density=True)
            ax.text(0.02, 0.95 - 0.05*(label+0.5), f"{name}: μ={np.mean(data):.3f}, σ={np.std(data):.3f}", 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)) 
        ax.set_xlabel(feature, fontweight='bold'); ax.set_ylabel('Density', fontweight='bold')
        ax.legend(fontsize=10, loc='upper right'); ax.grid(alpha=0.3, axis='y')
    axes[-1].axis('off')
    fig.suptitle(f'{title} - Feature Distributions', fontsize=14, fontweight='bold')
    return fig
    
feature_pdf_path = os.path.join(OUTPUT_PATH, "OcSVM_Feature_Distributions_AllFolds.pdf")
with PdfPages(feature_pdf_path) as pdf:
    # Page 1: Overall
    fig = plot_features(fold_id=None)
    pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(os.path.join(feature_dist_folder, "overall_feature_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    # Pages 2-6: Individual folds
    for fold_id in range(5):
        fig = plot_features(fold_id=fold_id)
        pdf.savefig(fig, bbox_inches='tight')
        fig.savefig(os.path.join(feature_dist_folder, f"fold{fold_id + 1}_feature_distributions.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    # Page 7: Statistics table
    fig, ax = plt.subplots(figsize=(16, 8)); ax.axis('off')
    stats_data = []
    for feat_idx, feature in enumerate(features):
        feat_col = X_cv_pool[:, feat_idx]
        nb = feat_col[y_cv_pool.values == 0]
        b = feat_col[y_cv_pool.values == 1]
        stats_data.append([feature] + 
                         [f"{np.mean(nb):.4f}", f"{np.std(nb):.4f}", f"{np.min(nb):.4f}", f"{np.max(nb):.4f}"] +
                         [f"{np.mean(b):.4f}", f"{np.std(b):.4f}", f"{np.min(b):.4f}", f"{np.max(b):.4f}"])
    table = ax.table(cellText=[['Feature', 'NB Mean', 'NB Std', 'NB Min', 'NB Max', 
                                'B Mean', 'B Std', 'B Min', 'B Max']] + stats_data, cellLoc='center', loc='center', colWidths=[0.12] + [0.11]*8)
    table.auto_set_font_size(False)
    table.set_fontsize(9); table.scale(1, 2)
    for i in range(9):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(stats_data) + 1):
        for j in range(9):
            table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else '#FFFFFF')
    fig.suptitle('Feature Statistics Summary', fontsize=14, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
print(f"[SUCCESS] PDF: {feature_pdf_path}")
print(f"[SUCCESS] PNG folder: {feature_dist_folder}")

# ========================================== HELPER FUNCTIONS ==========================================
def style_table(table, header_cols=None):
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    cells = table.get_celld()
    n_rows = max(row for (row, col) in cells.keys()) + 1
    n_cols = max(col for (row, col) in cells.keys()) + 1
    for col in range(n_cols):
        cells[(0, col)].set_facecolor('#34495E')
        cells[(0, col)].set_text_props(weight='bold', color='white')
    for row in range(1, n_rows):
        for col in range(n_cols):
            cells[(row, col)].set_facecolor('#ECF0F1' if row % 2 == 0 else '#FFFFFF')

def configure_ax(ax, xlabel='', ylabel='', title='', grid=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if grid:
        ax.grid(alpha=0.3, axis='y')
# ========================================== PRE-COMPUTE ALL STATISTICS ==========================================
print("[PREP] Pre-computing statistics and data structures...")
metrics_data = fold_df[['fold', 'recall', 'precision', 'f1', 'inner_cv_threshold']].copy()
recalls = metrics_data['recall'].values
precisions = metrics_data['precision'].values
f1s = metrics_data['f1'].values
thresholds = metrics_data['inner_cv_threshold'].values
score_data = {}
for fold_id in range(5):
    fold_samples = all_samples_results[all_samples_results['fold'] == fold_id]
    y_true_fold, scores_fold = [], []
    for _, row in fold_samples.iterrows():
        sample_id = row['sample_id']
        sample_subset = data[data['sample_id'] == sample_id]
        X_sample = sample_subset[features].values
        y_sample = sample_subset['Blast'].values
        scores_sample = final_model.decision_function(X_sample)
        y_true_fold.extend(y_sample)
        scores_fold.extend(scores_sample)
    if len(np.unique(y_true_fold)) > 1:
        fpr, tpr, _ = roc_curve(y_true_fold, -np.array(scores_fold))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_fold, -np.array(scores_fold))
        cm = confusion_matrix(y_true_fold, (-np.array(scores_fold) < final_threshold).astype(int))
        score_data[fold_id] = {
            'y_true': y_true_fold, 'scores': scores_fold,
            'fpr': fpr, 'tpr': tpr, 'roc_auc': auc(fpr, tpr),
            'precision': precision_curve, 'recall': recall_curve, 
            'pr_auc': auc(recall_curve, precision_curve), 'cm': cm}     
all_blast_scores = all_samples_results['mean_decision_score_blast'].dropna().values
all_nonblast_scores = all_samples_results['mean_decision_score_nonblast'].dropna().values

summary_metrics = {'recall': (recalls.mean(), recalls.std(), recalls.min(), recalls.max()),
    'precision': (precisions.mean(), precisions.std(), precisions.min(), precisions.max()),
    'f1': (f1s.mean(), f1s.std(), f1s.min(), f1s.max())}
score_stats = { 'nonblast': (np.mean(all_nonblast_scores), np.std(all_nonblast_scores), 
                 np.min(all_nonblast_scores), np.max(all_nonblast_scores)),
    'blast': (np.mean(all_blast_scores), np.std(all_blast_scores), 
              np.min(all_blast_scores), np.max(all_blast_scores))}
# ========================================== CREATE PDF ==========================================
pdf_path = os.path.join(OUTPUT_PATH, "OneClassSVM_Complete_Analysis.pdf")
with PdfPages(pdf_path) as pdf:
    # ========== PAGE 1: OVERALL PERFORMANCE SUMMARY ==========
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    # 1.1: Performance by Fold
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(5); width = 0.25
    for bars_data, offset, label, color in [
        (recalls, -width, 'Recall', '#FF6B6B'),
        (precisions, 0, 'Precision', '#4ECDC4'),
        (f1s, width, 'F1-Score', '#95E1D3')]:
        bars = ax1.bar(x + offset, bars_data, width, label=label, color=color, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    configure_ax(ax1, xlabel='Fold', ylabel='Score', title='One-Class SVM Performance Metrics by Fold')
    ax1.set_ylim([0, 1]); ax1.legend(fontsize=11)
    # 1.2 & 1.3: Distributions
    for idx, (data_arr, color, title_text) in enumerate([
        (recalls, '#FF6B6B', 'Recall'),
        (precisions, '#4ECDC4', 'Precision')]):
        ax = fig.add_subplot(gs[1, idx])
        ax.hist(data_arr, bins=10, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(data_arr.mean(), color=color, linestyle='--', linewidth=2, label=f'Mean: {data_arr.mean():.3f}')
        ax.axvline(np.median(data_arr), color='black', linestyle=':', linewidth=2, label=f'Median: {np.median(data_arr):.3f}')
        configure_ax(ax, xlabel=title_text, ylabel='Frequency', title=f'{title_text} Distribution')
        ax.legend(fontsize=9)
    # 1.4: Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    table_data = [['Metric', 'Mean', 'Std Dev', 'Min', 'Max']]
    for metric, (mean, std, min_v, max_v) in summary_metrics.items():
        table_data.append([metric.capitalize(), f'{mean:.3f}', f'{std:.3f}', f'{min_v:.3f}', f'{max_v:.3f}'])
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.2]*5)
    style_table(table)
    fig.suptitle('One-Class SVM - Overall Performance Summary', fontsize=16, fontweight='bold', y=0.995)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()
    
    # ========== PAGE 2: CONFUSION MATRICES FOR EACH FOLD ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices by Fold', fontsize=16, fontweight='bold')
    for fold_id in range(5):
        ax = axes.flatten()[fold_id]
        if fold_id in score_data:
            cm = score_data[fold_id]['cm']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, square=True, annot_kws={'size': 12})
            ax.set_title(f'Fold {fold_id + 1}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold'); ax.set_xlabel('Predicted Label', fontweight='bold')
    axes.flatten()[-1].axis('off'); plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()
    
    # ========== PAGES 3-4: ROC & PR CURVES (CONSOLIDATED) ==========
    for page_type, (curve_func, metric_name) in [
        ('ROC', (lambda fp, tp: (fp, tp, auc(fp, tp)), 'ROC Curves')),
        ('PR', (lambda p, r: (r, p, auc(r, p)), 'Precision-Recall Curves'))]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(metric_name + ' by Fold', fontsize=16, fontweight='bold')
        for fold_id in range(5):
            ax = axes.flatten()[fold_id]
            if fold_id in score_data:
                data_fold = score_data[fold_id]
                if page_type == 'ROC':
                    x, y, auc_val = data_fold['fpr'], data_fold['tpr'], data_fold['roc_auc']
                    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')
                else:
                    x, y, auc_val = data_fold['recall'], data_fold['precision'], data_fold['pr_auc']
                ax.plot(x, y, color='#FF6B6B', lw=2.5, label=f'AUC = {auc_val:.3f}')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.legend(loc="lower right" if page_type == 'ROC' else "upper right", fontsize=10)
                xlabel = 'False Positive Rate' if page_type == 'ROC' else 'Recall'
                ylabel = 'True Positive Rate' if page_type == 'ROC' else 'Precision'
                configure_ax(ax, xlabel=xlabel, ylabel=ylabel, title=f'Fold {fold_id + 1}')
        axes.flatten()[-1].axis('off'); plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close()
    
    # ========== PAGE 5: THRESHOLD ANALYSIS ==========
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    # 5.1: Thresholds by fold
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar([f'Fold {i+1}' for i in range(5)], thresholds, color='#95E1D3', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Threshold')
    for bar, thresh in zip(bars, thresholds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, f'{thresh:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    configure_ax(ax1, ylabel='Threshold Value', title='Optimal Decision Thresholds by Fold')
    ax1.legend(fontsize=11)
    # 5.2: Precision-Recall scatter
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(precisions, recalls, s=150, c=thresholds, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)
    plt.colorbar(scatter, ax=ax2, label='Threshold')
    configure_ax(ax2, xlabel='Precision', ylabel='Recall', title='Precision-Recall Trade-off')
    # 5.3: Threshold distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(thresholds, bins=10, color='#95E1D3', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(thresholds), color='teal', linestyle='--', linewidth=2, label=f'Mean: {np.mean(thresholds):.2f}')
    configure_ax(ax3, xlabel='Threshold Value', ylabel='Frequency', title='Threshold Distribution')
    ax3.legend()
    fig.suptitle('Threshold Analysis', fontsize=16, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()
    
    # ========== PAGE 6: DECISION FUNCTION SCORE DISTRIBUTION ==========
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    # 6.1: Overall distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_nonblast_scores, bins=30, alpha=0.6, label='Non-blast', color='#4ECDC4', edgecolor='black')
    ax1.hist(all_blast_scores, bins=30, alpha=0.6, label='Blast', color='#FF6B6B', edgecolor='black')
    configure_ax(ax1, xlabel='Decision Function Score', ylabel='Frequency', title='Decision Score Distribution (All Folds)')
    ax1.legend()
    # 6.2: Boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([all_nonblast_scores, all_blast_scores], labels=['Non-blast', 'Blast'], 
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#4ECDC4', '#FF6B6B']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    configure_ax(ax2, ylabel='Decision Function Score', title='Score Distribution by Class')
    # 6.3: Mean scores by fold
    ax3 = fig.add_subplot(gs[1, 0])
    fold_means_nb = all_samples_results.groupby('fold')['mean_decision_score_nonblast'].mean()
    fold_means_b = all_samples_results.groupby('fold')['mean_decision_score_blast'].mean()
    x = np.arange(5)
    ax3.bar(x - 0.175, fold_means_nb, 0.35, label='Non-blast', color='#4ECDC4', alpha=0.7)
    ax3.bar(x + 0.175, fold_means_b, 0.35, label='Blast', color='#FF6B6B', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xticks(x); ax3.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    configure_ax(ax3, xlabel='Fold', ylabel='Mean Decision Score', title='Mean Scores by Fold')
    ax3.legend()
    # 6.4: Statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    table_data = [['Metric', 'Non-blast', 'Blast']]
    for metric_name, metric_label in [('Mean', 0), ('Std', 1), ('Min', 2), ('Max', 3)]:
        table_data.append([metric_name, f'{score_stats["nonblast"][metric_label]:.3f}',
                          f'{score_stats["blast"][metric_label]:.3f}'])
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.33]*3)
    style_table(table)
    fig.suptitle('Decision Function Score Analysis', fontsize=16, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()
    
    # ========== PAGE 7: EXTERNAL TEST EVALUATION (COMPACT) ==========
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    # 7.1: ROC Curve (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr_ext, tpr_ext, color='#FF6B6B', lw=3, label=f'ROC-AUC = {metrics_ext["roc_auc"]:.3f}')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    ax1.fill_between(fpr_ext, tpr_ext, alpha=0.2, color='#FF6B6B')
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.05])
    configure_ax(ax1, xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve - External Test')
    ax1.legend(loc="lower right", fontsize=11)
    # 7.2: PR Curve (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(recall_curve_ext, precision_curve_ext, color='#4ECDC4', lw=3, label=f'PR-AUC = {metrics_ext["pr_auc"]:.3f}')
    ax2.fill_between(recall_curve_ext, precision_curve_ext, alpha=0.2, color='#4ECDC4')
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
    configure_ax(ax2, xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve - External Test')
    ax2.legend(loc="upper right", fontsize=11)
    # 7.3: Confusion Matrix (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm_external, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=True, square=True, annot_kws={'size': 16}, cbar_kws={'label': 'Count'})
    ax3.set_title('Confusion Matrix - External Test', fontsize=13, fontweight='bold')
    ax3.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax3.set_xticklabels(['Non-blast', 'Blast'], fontsize=11)
    ax3.set_yticklabels(['Non-blast', 'Blast'], rotation=0, fontsize=11)
    # 7.4:Performance Summary Table (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_table = [['Metric', 'Value'],
        ['Precision', f'{metrics_ext["precision"]:.3f}'],
        ['Recall', f'{metrics_ext["recall"]:.3f}'],
        ['F1-Score', f'{metrics_ext["f1"]:.3f}'],
        ['Accuracy', f'{metrics_ext["accuracy"]:.3f}'],
        ['ROC-AUC', f'{metrics_ext["roc_auc"]:.3f}'],
        ['PR-AUC', f'{metrics_ext["pr_auc"]:.3f}'],]
    table = ax4.table(cellText=summary_table, cellLoc='center', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    for i in range(1, 7):
        for j in range(2):
            table[(i, j)].set_facecolor('#E8F4F8' if i % 2 == 0 else '#FFFFFF')
    ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=15)
    fig.suptitle('External Test Evaluation - Unbiased Performance Assessment', fontsize=16, fontweight='bold', y=0.995)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()
        
print(f"[SUCCESS] PDF saved: {pdf_path}")

tot_sec = int(time.time() - script_start_time)
print("\n" + "="*20 + "PIPELINE EXECUTION COMPLETE" + "="*20)
print(f"Total Runtime: ({tot_sec//3600}:{(tot_sec % 3600) // 60}:{tot_sec % 60})")
print(f"Total Seconds: {tot_sec}")
