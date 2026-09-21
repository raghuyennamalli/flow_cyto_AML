import os
import glob
import pandas as pd
import numpy as np
from FlowCytometryTools import FCMeasurement
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
import re

# ==================== Paths ====================
healthy_folder = "/storage/mezya.sezen/mphasis/dataset/after_scaling/"
all_samples_folder = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
labels_folder = "/storage/mezya.sezen/mphasis/dataset/labels/"
output_folder = "/storage/mezya.sezen/mphasis/lof_results/lof_v3"
os.makedirs(output_folder, exist_ok=True)

features = ["SSC-A", "Horizon_V500-A", "PerCP-A", "PC7-A"]

# ==================== Helper functions ====================
def load_fcs_stable(file_path):
    sample = FCMeasurement(ID=file_path, datafile=file_path)
    df = sample.data.copy()

    try:
        meta_channels = sample.meta.get("_channels_", None)

        if meta_channels is not None and "$PnN" in meta_channels:
            instrument_cols = meta_channels["$PnN"].tolist()
            df.columns = [c.replace(" ", "_") for c in instrument_cols]
        else:
            print(f"[Warning] Using original column names for {file_path}")
            df.columns = [c.replace(" ", "_") for c in df.columns]

    except Exception as e:
        print(f"[Warning] Metadata read failed for {file_path}: {e}")
        df.columns = [c.replace(" ", "_") for c in df.columns]

    return df


def sample_events(df, n):
    return df.sample(n=n, random_state=42) if len(df) > n else df


# ==================== Step 1: Create aggregated 2K and 5K datasets ====================
if not os.path.exists(os.path.join(output_folder, "healthy_2K.pkl")):
    dataframes_2K = []
    dataframes_5K = []

    for f in glob.glob(os.path.join(healthy_folder, "*.fcs")):
        try:
            df = load_fcs_stable(f)
            sample_id = os.path.splitext(os.path.basename(f))[0]
            patient_id = "_".join(sample_id.split("_")[:2])
            df["patient_id"] = patient_id
            df["sample_id"] = sample_id

            df_2K = sample_events(df, 2000)
            df_5K = sample_events(df, 5000)

            dataframes_2K.append(df_2K)
            dataframes_5K.append(df_5K)
            print(df_2K.columns.unique())

        except Exception as e:
            print(f"Error {f}: {e}")

    pd.concat(dataframes_2K).reset_index(drop=True).to_pickle(os.path.join(output_folder, "healthy_2K.pkl"))
    pd.concat(dataframes_5K).reset_index(drop=True).to_pickle(os.path.join(output_folder, "healthy_5K.pkl"))
    print("[OK] Downsampled 2K and 5K pickles created.")


# ==================== Step 2: Use 2K for cross-validation ====================
df_2K = pd.read_pickle(os.path.join(output_folder, "healthy_2K.pkl"))
X_2K = df_2K[[c for c in features if c in df_2K.columns]].values
groups_2K = df_2K["patient_id"].values


class LOFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=20, contamination=0.01):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X, y=None):
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True
        )
        self.model_.fit(X)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        return np.where(preds == -1, 1, 0)  # 1 = anomaly ("blast"), 0 = normal


# ---------------- ADDING CUSTOM SCORING -----------------

# ---------------- FIXED CUSTOM SCORING -----------------

def lof_score(estimator, X, y=None):
    preds = estimator.predict(X)
    return -np.mean(preds)


# ---------------------------------------------------------

param_grid = {
    "n_neighbors": [10, 20, 50, 100],
    "contamination": [0.01, 0.025, 0.05, 0.1]
}

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=3)
cv_results = []


for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_2K, np.zeros(len(X_2K)), groups_2K)):
    print(f"\n=== Outer Fold {fold+1} ===")

    X_train, X_val = X_2K[train_idx], X_2K[val_idx]
    groups_train = groups_2K[train_idx]

    gs = GridSearchCV(
        LOFWrapper(),
        param_grid,
        cv=inner_cv.split(X_train, np.zeros(len(X_train)), groups_train),
        scoring = lof_score,   # <-- FIXED HERE
        refit=True
    )

    gs.fit(X_train)
    best_model = gs.best_estimator_

    y_val_pred = best_model.predict(X_val)
    val_pred_mean = np.mean(y_val_pred)
    print(f"Fraction classified as 'blast' in healthy validation: {val_pred_mean:.4f}")

    cv_results.append({
        "fold": fold+1,
        "best_n_neighbors": gs.best_params_["n_neighbors"],
        "best_contamination": gs.best_params_["contamination"],
        "healthy_false_positive_rate": val_pred_mean
    })


final_n_neighbors = Counter([r['best_n_neighbors'] for r in cv_results]).most_common(1)[0][0]
final_contamination = Counter([r['best_contamination'] for r in cv_results]).most_common(1)[0][0]

print(f"\nBest (majority) params from CV: n_neighbors={final_n_neighbors}, contamination={final_contamination}")


# ==================== Step 3: Train final LOF on 5K ====================
df_5K = pd.read_pickle(os.path.join(output_folder, "healthy_5K.pkl"))
X_5K = df_5K[[c for c in features if c in df_5K.columns]].values

final_lof = LOFWrapper(n_neighbors=final_n_neighbors, contamination=final_contamination)
final_lof.fit(X_5K)


# ==================== Step 4: Evaluate on AML test samples ====================
precisions, recalls, f1s = [], [], []

for fcs_file in glob.glob(os.path.join(all_samples_folder, "*.fcs")):
    file_base = os.path.basename(fcs_file)

    # Extract sample key using regex: matches BLAST110_41_P1 from BLAST110_41_P1_Cleaned_transformed_scaled.fcs
    m = re.match(r"(BLAST\d+_\d+_P\d+)", file_base)
    if m:
        key = m.group(1)
    else:
        print(f"[SKIP] Could not extract key from {file_base}")
        continue

    pattern = os.path.join(labels_folder, f"{key}.csv")
    matching = glob.glob(pattern)
    if not matching:
        print(f"[SKIP] No label match for {key}")
        continue
    
    labels_df = pd.read_csv(matching[0])
    df_test = load_fcs_stable(fcs_file)
    df_test = sample_events(df_test, 5000)
    X_test = df_test[[c for c in features if c in df_test.columns]].values


    y_true = labels_df["Blast"].values[:len(df_test)]
    y_pred = final_lof.predict(X_test)

    # --- DEBUG PRINTS ---
    print(f"\n--- {fcs_file} ---")
    print("Number of test cells:", len(df_test))
    print("Number of predicted blasts:", np.sum(y_pred))
    print("Number of true blasts:", np.sum(y_true))
    print("Unique values in y_pred:", np.unique(y_pred, return_counts=True))
    print("Unique values in y_true:", np.unique(y_true, return_counts=True))
    # -------------------

    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred, zero_division=0))
    f1s.append(f1_score(y_true, y_pred, zero_division=0))

print("\n=== Final evaluation on AML test samples ===")
print("Precision:", np.mean(precisions))
print("Recall:", np.mean(recalls))
print("F1:", np.mean(f1s))




# ==================== Step 5: Save CV and test results ====================
pd.DataFrame(cv_results).to_csv(os.path.join(output_folder, "lof_cv_healthy_results.csv"), index=False)
pd.DataFrame([{
    "precision": np.mean(precisions),
    "recall": np.mean(recalls),
    "f1": np.mean(f1s)
}]).to_csv(os.path.join(output_folder, "lof_final_metrics.csv"), index=False)
