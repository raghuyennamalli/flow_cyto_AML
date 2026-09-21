import os
import time
import shap
import numpy as np
import pandas as pd
from FlowCytometryTools import FCMeasurement
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score,roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, make_scorer, precision_recall_curve, roc_curve, auc)
from joblib import parallel_backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from collections import Counter
import json 
from scipy import stats

# ========================================== PARAMETERS & PATHS ==========================================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/Changed_para/"
os.makedirs(os.path.join(OUTPUT_PATH, "RF_CV"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "RF_Plots"), exist_ok=True)

features = ["SSC-A","Horizon V450-A","Horizon V500-A", "PerCP-A", "PC7-A"]

# ============================== HELPER FUNCTIONS ===============================
def from_fcs(path):
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()
        
def find_optimal_threshold(y_true, y_scores):      
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    thresholds = np.append(0, thresholds)
    optimal_thresh = thresholds[ix] 
    y_pred = (y_scores >= optimal_thresh).astype(int)
    metrics = { 'threshold': optimal_thresh,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)}
    return optimal_thresh, metrics
 
def create_kde_plots_for_fold(fold, X_train, y_train, X_test, y_test, features, output_path):
    print(f"\n[CREATING KDE PLOTS FOR FOLD {fold}]")
    train_df = X_train.copy()
    train_df['Label'] = y_train.values
    test_df = X_test.copy()
    test_df['Label'] = y_test.values
    for feature in features:
        plt.figure(figsize=(12, 6))
        # Train distributions
        train_blast = train_df[train_df['Label'] == 1][feature]
        train_nonblast = train_df[train_df['Label'] == 0][feature]
        # Test distributions
        test_blast = test_df[test_df['Label'] == 1][feature]
        test_nonblast = test_df[test_df['Label'] == 0][feature]
        # Plot KDE curves
        sns.kdeplot(train_nonblast, color='blue', linewidth=2.5, linestyle='-', label='Train Non-blast', fill=True, alpha=0.3)
        sns.kdeplot(train_blast, color='red', linewidth=2.5, linestyle='-', label='Train Blast', fill=True, alpha=0.3)
        sns.kdeplot(test_nonblast, color='blue', linewidth=2.5, linestyle='--', label='Test Non-blast')
        sns.kdeplot(test_blast, color='red', linewidth=2.5, linestyle='--', label='Test Blast')
        plt.xlabel(feature, fontsize=12); plt.ylabel('Density', fontsize=12)
        plt.title(f'Fold {fold}: {feature} - Train vs Test Distribution', fontsize=14, fontweight='bold')
        plt.legend(loc='best');plt.grid(True, alpha=0.3)
        plot_name = f"fold{fold}_{feature.replace(' ', '_').replace('-', '_')}_distribution.png"
        plot_path = os.path.join(output_path, "RF_Plots", plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved KDE Plots")
        
# ========= STEP 0: CREATE AGGREGATED DATASET (2K + 5K) ===========================
pkl_2k = os.path.join(OUTPUT_PATH, "BLAST110_2K.pkl")
pkl_5k = os.path.join(OUTPUT_PATH, "BLAST110_5K.pkl")

'''if not os.path.exists(pkl_5k):
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
            ff = ff[(ff["Singlets"] == 1)]
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
    print("[INFO] Pickled data already exists.")'''
# ================= STEP 1: LOAD AGGREGATED DATA =========================
pkl_5k = os.path.join(OUTPUT_PATH, "BLAST110_5K.pkl")
data = pd.read_pickle(pkl_5k)
X = data[features]
y = data["Blast"]
groups = data["patient_id"]
samples = data["sample_id"]
X_scaled= X.values
print(f"[INFO] Data shape: {X_scaled.shape}")
print(f"[INFO] Blast prevalence: {y.mean():.3%}")

# ================== STEP 2: DEFINE INNER + OUTER CV ==================================
outer_cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
inner_cv = StratifiedGroupKFold(n_splits=5)

param_grid = { "max_depth": [10,20,30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1,2, 4,8],
    "n_estimators": [50, 100, 200, 300]}
    
"""params = {"max_depth" : range(2, 11),
              "min_samples_split": range(100, 10001),
              "min_samples_leaf": range(100, 10001)}"""

# ======== STEP 3:  CV LOOP  (WITH PARALLELIZATION) =======================
total_start_time = time.time()
threshold_results = []
all_fold_results = []
external_results = []
best_models = []
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y, groups)):
    print(f"\n{'='*20}OUTER FOLD {fold+1}/{outer_cv.get_n_splits()} {'='*20}")
   
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

    # ------------------------ INNER CV (Grid Search)------------------------
    print(f"\n[INNER CV] Starting grid search with parallelization...")
    base_rf = RandomForestClassifier(random_state=fold, n_jobs=1)
    inner_cv_start = time.time() 
    grid = GridSearchCV(base_rf,
            param_grid,
            cv=inner_cv,
            scoring='average_precision',
            n_jobs=-1,
            refit=True, verbose=0)        
    start = time.time()
    grid.fit(X_train, y_train, groups=train_groups)
    elapsed_inner = time.time() - start
    inner_cv_elapsed = time.time() - inner_cv_start
    
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_RandomForest_innerCV.csv"), index=False)
    
    best_rf = grid.best_estimator_
    best_params = grid.best_params_
    best_models.append({'fold': fold, 'model': best_rf, 'params': grid.best_params_})
    
    print(f"[INNER CV] Complete ({elapsed_inner:.1f}s)")
    print(f"[INNER CV] Best Score: {grid.best_score_:.4f}")
    print(f"[INNER CV] Best Params: {grid.best_params_}")
    
    # ------------------------ THRESHOLD OPTIMIZATION (VALIDATION FOLD) -------
    train_probas_cv = cross_val_predict(best_rf, X_train, y_train, cv=inner_cv, groups=train_groups, method='predict_proba', n_jobs=-1)[:, 1]
    print(f"[THRESHOLD] Determining optimal threshold on Training data (CV)...")
    optimal_thresh_val, train_thresh_metrics = find_optimal_threshold(y_train.values, train_probas_cv)
    
    # ------------------------ OUTER TESTING ------------------------
    outer_test_start = time.time()
    proba_scores_test = best_rf.predict_proba(X_test)[:, 1]
    preds = (proba_scores_test >= optimal_thresh_val).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)  
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    outer_test_elapsed = time.time() - outer_test_start

    print(f"\n[OUTER CV] Confusion Matrix: \n {confusion_matrix(y_test, preds)}")
    print(f"\n[OUTER CV] Classification Report:\n {classification_report(y_test, preds, target_names=['Non-Blast', 'Blast'])}")
    print(f"\n[THRESHOLD ANALYSIS]")
    print(f"Optimal threshold: {optimal_thresh_val:.4f}")
    print(f"Accuracy:{acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    threshold_results.append({'fold': fold,'best_params': str(grid.best_params_),
        'optimal_threshold': optimal_thresh_val,
        'precision': prec,'recall': rec,'f1': f1, 'accuracy': acc,
        'runtime_seconds': elapsed_inner,'inner_cv_time': inner_cv_elapsed,
        'outer_test_time': outer_test_elapsed })
    predictions_data = {'fold': fold,
     'y_true': y_test.values,
     'proba_scores': proba_scores_test,
     'optimal_threshold': optimal_thresh_val }
    pred_path = os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_predictions.pkl")
    
    with open(pred_path, 'wb') as f:
        pickle.dump(predictions_data, f)
    print(f"[SAVED] Predictions: {pred_path}")
    
    # ------------------------ KDE PLOTS FOR THIS FOLD ------------------------
    X_train_df = pd.DataFrame(X_train, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)
    create_kde_plots_for_fold(fold=fold, X_train=X_train_df, y_train=y_train.reset_index(drop=True), X_test=X_test_df, y_test=y_test.reset_index(drop=True), features=features, output_path=OUTPUT_PATH)
    # ------------------------ PER-SAMPLE RESULTS ------------------------
    per_sample_start = time.time()
    sample_ids = samples.iloc[test_idx].unique()
    model_results = []

    for sid in sample_ids:
        subset = data[data["sample_id"] == sid]
        X_sub = subset[features].values
        y_sub = subset["Blast"].values
        proba_sub = best_rf.predict_proba(X_sub)[:, 1]
        pred_sub = (proba_sub >= optimal_thresh_val).astype(int)

        result_dict = {  "fold": fold,
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
            "mean_proba_nonblast": np.mean(proba_sub[y_sub == 0]) if np.any(y_sub == 0) else np.nan,
            "optimal_threshold": optimal_thresh_val}
        model_results.append(result_dict)
    per_sample_elapsed = time.time() - per_sample_start
    pd.DataFrame(model_results).to_csv(os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_RandomForest_outerCV.csv"),index=False)
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

# ========================================== STEP 4: FINAL MODEL TRAINING (on 100% of data) ==========================================
print("\n" + "="*30 + "[FINAL MODEL TRAINING] Training on entire labeled dataset..." + "="*30)
print("\nSelecting best hyperparameters from nested CV...")
best_params_all_folds = [bm['params'] for bm in best_models]

param_counter = Counter([tuple(sorted(p.items())) for p in best_params_all_folds])
best_param_tuple = param_counter.most_common(1)[0][0]
best_params = dict(best_param_tuple)

print("Best hyperparameters selected from nested CV:")
print(best_params)
final_start = time.time()
avg_threshold = np.mean([r['optimal_threshold'] for r in threshold_results])
print(f"Final threshold: {avg_threshold:.4f}")
final_rf = RandomForestClassifier(random_state=42,n_jobs=-1,**best_params)
final_rf.fit(X_scaled, y)
 
final_model_path = os.path.join(OUTPUT_PATH, "RF_final_model.pkl")
with open(final_model_path, 'wb') as f:
    pickle.dump(final_rf, f)
final_training_time = time.time() - final_start
print(f"Final model trained in {final_training_time:.1f}s")
print(f"Final model: {final_rf}")

print(f"\n[SAVED] Final model: {final_model_path}")
final_config = {
    'features': features,
    'best_hyperparameters': best_params,
    'optimal_threshold': float(avg_threshold),
    'training_samples': len(X_scaled),
    'blast_prevalence': float(y.mean()),
    'cv_f1_estimate': float(np.mean([r['f1'] for r in threshold_results])),  # ← ADD THIS
    'final_training_time': final_training_time}

config_path = os.path.join(OUTPUT_PATH, "RF_CV", "final_model_config.json")
with open(config_path, 'w') as f:
    json.dump(final_config, f, indent=2)
print(f"[SAVED] Model configuration: {config_path}")

# =========== STEP 4b: FEATURE IMPORTANCE PLOTS ===========
print("\n" + "="*30 + "[GENERATING FEATURE IMPORTANCE PLOTS]" + "="*30)
feature_importances = final_rf.feature_importances_
feature_names = features
sorted_idx = np.argsort(feature_importances)[::-1]
# Plot: Feature Importance Bar Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], color='steelblue')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Feature Importance in Final Random Forest Model', fontsize=14, fontweight='bold')
plt.tight_layout()
importance_path = os.path.join(OUTPUT_PATH, "RF_Plots", "final_model_feature_importance.png")
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
print(f"[SAVED] Feature importance: {importance_path}")
plt.close()
print("\nFeature Importance Ranking:")
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1}. {feature_names[idx]}: {feature_importances[idx]:.4f}")

# ========================================STEP 6: SAVE FINAL SUMMARIES ========================================================================================
pd.DataFrame(threshold_results).to_csv(os.path.join(OUTPUT_PATH, "RF_CV", "threshold_summary.csv"),index=False)
pd.DataFrame(all_fold_results).to_csv(os.path.join(OUTPUT_PATH, "RF_CV", "all_samples_results.csv"),index=False)

# ========= STEP 7a: ROC & PR-AUC CURVES FOR EACH FOLD WITH 95% CI ==========
print("\n" + "="*20 + "[GENERATING ROC AND PR-AUC CURVES" + "="*20)
roc_auc_scores = []
pr_auc_scores = []
for fold in range(5):
    pred_path = os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_predictions.pkl")    
    try:
        with open(pred_path, 'rb') as f:
            pred_data = pickle.load(f)
        y_test = pred_data['y_true']
        proba_scores = pred_data['proba_scores']
        # ===== ROC CURVE =====
        fpr, tpr, _ = roc_curve(y_test, proba_scores)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
        plt.title(f'ROC Curve - Fold {fold}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11); plt.grid(True, alpha=0.3)
        roc_path = os.path.join(OUTPUT_PATH, "RF_Plots", f"fold{fold}_roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Fold {fold}: ROC-AUC saved")
        # ===== PR-AUC CURVE =====
        precision, recall, _ = precision_recall_curve(y_test, proba_scores)
        pr_auc = auc(recall, precision)
        pr_auc_scores.append(pr_auc)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='red', lw=2.5, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=np.mean(y_test), color='gray', linestyle='--', lw=2, alpha=0.7, label=f'No Skill ({np.mean(y_test):.3f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - Fold {fold}', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=11); plt.grid(True, alpha=0.3)
        pr_path = os.path.join(OUTPUT_PATH, "RF_Plots", f"fold{fold}_precision_recall_curve.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Fold {fold}: PR-AUC saved")
    except FileNotFoundError:
        print(f"Fold {fold}: Predictions file not found")
        pass
# ========= STEP 7b: CONFUSION MATRICES AS HEATMAPS ==========
print("\n" + "="*30 +"[GENERATING CONFUSION MATRIX HEATMAPS]"+ "="*30)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for fold in range(5):
    ax = axes[fold]
    pred_path = os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_predictions.pkl")
    try:
        with open(pred_path, 'rb') as f:
            pred_data = pickle.load(f)
        y_test = pred_data['y_true']
        proba_scores = pred_data['proba_scores']
        optimal_threshold = pred_data['optimal_threshold']
        
        y_pred = (proba_scores >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', xticklabels=['Non-blast', 'Blast'], yticklabels=['Non-blast', 'Blast'],cbar_kws={'label': 'Normalized Count'}, ax=ax, annot_kws={"size": 11})
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_title(f'Fold {fold}: Confusion Matrix', fontweight='bold', fontsize=11)
        
        print(f"Fold {fold}: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")        
    except FileNotFoundError:
        print(f"Fold {fold}: Predictions file not found")
        pass
        
fig.delaxes(axes[-1])
plt.suptitle('Confusion Matrices: All Folds', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_PATH, "RF_Plots", "confusion_matrices_all_folds.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] Confusion matrices: {cm_path}")
plt.close()

# ========= STEP 7c: PROBABILITY DISTRIBUTION PLOTS ==========
print("\n" + "="*30+ "[GENERATING PROBABILITY DISTRIBUTION PLOTS]" + "="*30)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for fold in range(5):
    ax = axes[fold]
    pred_path = os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_predictions.pkl") 
    try:
        with open(pred_path, 'rb') as f:
            pred_data = pickle.load(f)
        y_test = pred_data['y_true']
        proba_scores = pred_data['proba_scores']
        optimal_threshold = pred_data['optimal_threshold']      
        # Separate by class
        nonblast_proba = proba_scores[y_test == 0]
        blast_proba = proba_scores[y_test == 1]      
        # Plot histograms
        ax.hist(nonblast_proba, bins=50, alpha=0.6, label='Non-blast', color='blue', density=True)
        ax.hist(blast_proba, bins=50, alpha=0.6, label='Blast', color='red', density=True)
        ax.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2.5,label=f'Threshold={optimal_threshold:.3f}')
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Fold {fold}: Probability Distribution', fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Print separation stats
        mean_nonblast = np.mean(nonblast_proba)
        mean_blast = np.mean(blast_proba)
        separation = mean_blast - mean_nonblast
        print(f"Fold {fold}: Mean Non-blast prob={mean_nonblast:.3f}, Mean Blast prob={mean_blast:.3f}, Separation={separation:.3f}")       
    except FileNotFoundError:
        print(f"Fold {fold}: Predictions file not found")
        pass
fig.delaxes(axes[-1])
plt.suptitle('Predicted Probability Distributions: All Folds', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
prob_path = os.path.join(OUTPUT_PATH, "RF_Plots", "probability_distributions_all_folds.png")
plt.savefig(prob_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] Probability distributions: {prob_path}")
plt.close()

# ========= STEP 7d: PERFORMANCE METRICS ACROSS FOLDS ==========
print("\n" + "="*30+"[GENERATING PERFORMANCE SUMMARY PLOTS]"+ "="*30)

avg_results = pd.DataFrame(threshold_results)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# Plot 1: F1 Scores across folds
axes[0, 0].bar(range(5), avg_results['f1'], color='steelblue', alpha=0.8, edgecolor='black')
axes[0, 0].axhline(avg_results['f1'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {avg_results['f1'].mean():.3f}")
for i, v in enumerate(avg_results['f1']):
    axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
axes[0, 0].set_ylabel('F1 Score', fontsize=11)
axes[0, 0].set_xlabel('Fold', fontsize=11)
axes[0, 0].set_title('F1 Scores Across Folds', fontweight='bold', fontsize=12)
axes[0, 0].set_ylim([min(avg_results['f1']) - 0.02, max(avg_results['f1']) + 0.02])
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Precision vs Recall
axes[0, 1].scatter(avg_results['recall'], avg_results['precision'], s=200, alpha=0.6, color='darkgreen', edgecolor='black')
for i, fold in enumerate(avg_results['fold']):
    axes[0, 1].annotate(f'F{int(fold)}', (avg_results.loc[i, 'recall'], avg_results.loc[i, 'precision']),fontsize=10, ha='center')
axes[0, 1].set_xlabel('Recall', fontsize=11); axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Precision vs Recall Across Folds', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: All metrics comparison
metrics = ['precision', 'recall', 'f1']
x = np.arange(len(metrics))
width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, fold in enumerate(avg_results['fold']):
    values = [avg_results.loc[i, m] for m in metrics]
    axes[1, 0].bar(x + i*width, values, width, label=f'Fold {int(fold)}', alpha=0.8, color=colors[i])
axes[1, 0].set_ylabel('Score', fontsize=11)
axes[1, 0].set_title('Metrics Comparison Across Folds', fontweight='bold', fontsize=12)
axes[1, 0].set_xticks(x + width * 2); axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([min(avg_results[['precision', 'recall', 'f1']].min()) - 0.02, 1.0])

# Plot 4: Optimal thresholds distribution
axes[1, 1].plot(avg_results['fold'], avg_results['optimal_threshold'], marker='o', linewidth=2.5, markersize=10, color='purple', markeredgecolor='black', markeredgewidth=1.5)
axes[1, 1].axhline(avg_results['optimal_threshold'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {avg_results['optimal_threshold'].mean():.3f}")
for i, (fold, thresh) in enumerate(zip(avg_results['fold'], avg_results['optimal_threshold'])):
    axes[1, 1].text(fold, thresh + 0.005, f'{thresh:.3f}', ha='center', va='bottom', fontsize=9)
axes[1, 1].set_ylabel('Optimal Threshold', fontsize=11)
axes[1, 1].set_xlabel('Fold', fontsize=11)
axes[1, 1].set_title('Optimal Thresholds Across Folds', fontweight='bold', fontsize=12)
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Cross-Validation Performance Summary', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
summary_path = os.path.join(OUTPUT_PATH, "RF_Plots", "cv_performance_summary.png")
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVED] Performance summary: {summary_path}")
plt.close()

print(f"\n[SUMMARY STATISTICS]")
print(f"Accuracy: {avg_results['accuracy'].mean():.3f} + or - {avg_results['accuracy'].std():.3f}")
print(f"Precision: {avg_results['precision'].mean():.3f} ± {avg_results['precision'].std():.3f}")
print(f"Recall:    {avg_results['recall'].mean():.3f} ± {avg_results['recall'].std():.3f}")
print(f"F1 Score:  {avg_results['f1'].mean():.3f} ± {avg_results['f1'].std():.3f}")
print(f"Threshold: {avg_results['optimal_threshold'].mean():.3f} ± {avg_results['optimal_threshold'].std():.3f}")

# ========================================== STEP 7: DIAGNOSTICS: Threshold Curve Analysis ==========================================
print("\n" + "="*30 +"THRESHOLD ANALYSIS: Precision/Recall/F1 Curves Across All Folds " + "="*30)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for fold in range(5):
    ax = axes[fold]
    pred_path = os.path.join(OUTPUT_PATH, "RF_CV", f"fold{fold}_predictions.pkl")
    try:
        with open(pred_path, 'rb') as f:
            data = pickle.load(f)
        y_test = data['y_true']
        proba_scores = data['proba_scores']
        optimal_threshold = data['optimal_threshold']
        #precision-recall-F1 curves
        precision, recall, thresholds = precision_recall_curve(y_test, proba_scores)
        f1s = 2 * precision * recall / (precision + recall + 1e-8)
        # Plot curves
        ax.plot(thresholds, precision[:-1], label="Precision", linewidth=2.5, color='blue')
        ax.plot(thresholds, recall[:-1], label="Recall", linewidth=2.5, color='green')
        ax.plot(thresholds, f1s[:-1], label="F1", linewidth=2.5, color='purple')
        # Mark optimal threshold
        ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2.5, label=f'Optimal: {optimal_threshold:.3f}')
        # Get corresponding metrics at optimal threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        ax.set_title(f"Fold {fold}: Threshold Analysis\nOptimal F1: {f1s[idx]:.3f}", fontweight='bold', fontsize=11)
        ax.set_xlabel("Threshold", fontsize=10); ax.set_ylabel("Metric Value", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.3)
        print(f"\n Fold {fold}: data available and plotted")
        
        #individual plot
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2.5, marker='o', markersize=3)
        plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2.5, marker='s', markersize=3)
        plt.plot(thresholds, f1s[:-1], label="F1", linewidth=2.5, marker='^', markersize=3)
        plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2.5, label=f'Optimal Threshold: {optimal_threshold:.3f}')
        plt.fill_between(thresholds, 0, 1, alpha=0.1, color='gray')
        plt.title(f"Fold {fold}: Detailed Threshold Analysis", fontweight='bold', fontsize=12)
        plt.xlabel("Decision Threshold", fontsize=11); plt.ylabel("Metric Value", fontsize=11)
        plt.legend(fontsize=10, loc='best'); plt.grid(True, alpha=0.3)
        plt.xlim(0, 1); plt.ylim(0, 1.05)
        individual_path = os.path.join(OUTPUT_PATH, "RF_Plots", f"fold{fold}_threshold_curve_detailed.png")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"Saved")
        plt.close()  
    except FileNotFoundError:
        pass
fig.delaxes(axes[-1])
plt.suptitle("Threshold Analysis: Precision/Recall/F1 Across All 5 Folds",fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
save_path = os.path.join(OUTPUT_PATH, "RF_Plots", "threshold_analysis_all_folds.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n Saved\n")
plt.close()

# ==========================================STEP 8:  FINAL RUNTIME REPORT ==========================================
total_elapsed = time.time() - total_start_time
print("\n" + "="*70 + "NESTED CROSS-VALIDATION COMPLETE" + "="*70)
print(f"\n[RUNTIME SUMMARY]")
print(f"Total runtime:{total_elapsed/3600:.2f} hours or {total_elapsed/60:.1f} minutes ({total_elapsed:.1f} seconds)")

avg_results = pd.DataFrame(threshold_results)
print(f"\n[PER-FOLD BREAKDOWN]")
print(f"Average total fold time: {avg_results['total_fold_time'].mean():.1f}s")
print(f"  - Inner CV: {avg_results['inner_cv_time'].mean():.1f}s")
print(f"  - Outer test: {avg_results['outer_test_time'].mean():.1f}s")
print(f"  - Per-sample: {avg_results['per_sample_time'].mean():.1f}s")
print(f"\n[PERFORMANCE SUMMARY]")
print(f"Average Accuracy: {avg_results['accuracy'].mean():.3f} + or - {avg_results['accuracy'].std():.3f}")
print(f"Average Precision: {avg_results['precision'].mean():.3f} + or - {avg_results['precision'].std():.3f}")
print(f"Average Recall: {avg_results['recall'].mean():.3f}  +/-{avg_results['recall'].std():.3f}")
print(f"Average F1: {avg_results['f1'].mean():.3f} +\- {avg_results['f1'].std():.3f}")
print(f"\n OUTPUT FILES saved")