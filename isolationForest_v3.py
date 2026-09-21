import os
import time
import numpy as np
import pandas as pd
from FlowCytometryTools import FCMeasurement
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample
from itertools import product 



import sys
print(sys.version)

start_time = time.time()
print(f"[DEBUG] Script started at: {time.ctime(start_time)}")

# -------------------------------
# Parameters
# -------------------------------
fcs_folder = "/storage/mezya.sezen/mphasis/dataset/after_scaling/"   # where your .fcs files live
labels_folder = "/storage/mezya.sezen/mphasis/dataset/Labels"              # folder with label csvs


# -------------------------------
# Helper: assign labels
# -------------------------------
def assign_label(row):
    if row.get("Blast", 0) == 1:
        return "Blast"
    elif row.get("/WBC/Singlets/CD45pos/Lymphocytes", 0) == 1:
        return "Lymphocyte"
    elif row.get("WBC", 0) == 1:
        return "WBC"
    else:
        return "Other"

# -------------------------------
# Step 1: Load all data
# -------------------------------
all_data = []
common_features = None

for fname in os.listdir(fcs_folder):
    if fname.endswith("_cleaned_doubletRemoval_transformed_scaled.fcs"):
        fpath = os.path.join(fcs_folder, fname)
        label_base = fname.replace("_cleaned_doubletRemoval_transformed_scaled.fcs", "")
        label_path = os.path.join(labels_folder, f"{label_base}.csv")

        if not os.path.exists(label_path):
            continue

        sample = FCMeasurement(ID=label_base, datafile=fpath)
        features_df = sample.data.copy()

        if common_features is None:
            common_features = set(features_df.columns)
        else:
            common_features &= set(features_df.columns)

        labels_df = pd.read_csv(label_path)
        labels_df["Final_Label"] = labels_df.apply(assign_label, axis=1)

        merged = features_df.merge(
            labels_df[["event_ID", "Final_Label"]],
            on="event_ID", how="inner"
        )
        all_data.append(merged)

# Combine and keep only shared features
data = pd.concat(all_data, ignore_index=True)
features = ["SSC-A", "Horizon V500-A", "PerCP-A", "PC7-A"]
X = data[features].copy()
y = data["Final_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separate normal (WBC) and abnormal (Blast)
X_wbc = X_scaled[y == "WBC"]
X_blast = X_scaled[y == "Blast"]

# -------------------------------
# Step: Split WBC for training/val/test
# -------------------------------
from sklearn.utils import resample

# -------------------------------
# Train 60% of WBC
# -------------------------------
X_train_wbc, X_remaining_wbc = train_test_split(X_wbc, test_size=0.4, random_state=42)

# Remaining 40% WBC + all Blast for val/test
X_remaining_blast = X_blast.copy()

# Create validation set
X_val_wbc, X_test_wbc = train_test_split(X_remaining_wbc, test_size=0.5, random_state=42)
X_val_blast, X_test_blast = train_test_split(X_remaining_blast, test_size=0.5, random_state=42)

# Ensure balanced sets
n_val = min(len(X_val_wbc), len(X_val_blast))
X_val_wbc = X_val_wbc[:n_val]
X_val_blast = X_val_blast[:n_val]

X_val = np.vstack([X_val_wbc, X_val_blast])
y_val = np.array([1]*len(X_val_wbc) + [-1]*len(X_val_blast))

# Similar approach for test set
n_test = min(len(X_test_wbc), len(X_test_blast))
X_test_wbc = X_test_wbc[:n_test]
X_test_blast = X_test_blast[:n_test]

X_test = np.vstack([X_test_wbc, X_test_blast])
y_test = np.array([1]*len(X_test_wbc) + [-1]*len(X_test_blast))

# Training set: only 60% WBC
X_train = X_train_wbc

print(f"[DEBUG] Train WBC: {len(X_train)}")
print(f"[DEBUG] Validation set -> WBC: {sum(y_val==1)}, Blast: {sum(y_val==-1)}")
print(f"[DEBUG] Test set -> WBC: {sum(y_test==1)}, Blast: {sum(y_test==-1)}")

# -------------------------------
# Step 2: IsolationForest training
# -------------------------------
print("\n=== Isolation Forest for Blast Detection ===\n")
results=[]
best_f1=0
param_grid = {
    "n_estimators": [100],
    "max_samples": [0.6, 0.8],
	"contamination": [0.001],
    "random_state": [42]
	
}

# Generate all parameter combinations
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_samples"],
	 param_grid["contamination"],
    param_grid["random_state"]
))

for n_estimators, max_samples, contamination, random_state in param_combinations:
    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
		 "contamination": contamination,
        "random_state": random_state
    }

    print(f"[DEBUG] Testing Parameters: {params}")
    start_time = time.time()

    model = IsolationForest(**params, n_jobs=8)
    model.fit(X_train)
    # Evaluate
    preds_val = model.predict(X_val)

    print("\n[VALIDATION RESULTS]")
    print(confusion_matrix(y_val, preds_val, labels=[1, -1]))
    print(classification_report(
        y_val, preds_val, target_names=["Normal(WBC)", "Blast/Outlier"]
    ))

    elapsed = time.time() - start_time
    print(f"[DEBUG] Runtime: {elapsed:.1f}s")
    print("-" * 60)

    
    f1 = f1_score(y_val, preds_val, pos_label=-1)  # -1 for anomalies
    precision = precision_score(y_val, preds_val, pos_label=-1)
    recall = recall_score(y_val, preds_val, pos_label=-1)
    
    results.append({
        'params': params,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_params = params
        
print(f"\nBest Parameters: {best_params}")
print(f"Best F1 Score: {best_f1:.4f}")


# -------------------------------
# Step 3: Evaluate on complete FCS files (external test)
# -------------------------------
print("\n=== External Test on Full FCS Files ===\n")

# Define test FCS files (use actual filenames present in your folder)
external_test_files = [
    "/storage/mezya.sezen/mphasis/dataset/final_scaled/BLAST110_13_P3_Cleaned_transformed_scaled.fcs",
    
]

# Loop through each external file
for test_fname in external_test_files:
    test_path = os.path.join(fcs_folder, test_fname)
    label_base = test_fname.replace("_cleaned_doubletRemoval_transformed_scaled.fcs", "")
    label_path = os.path.join(labels_folder, f"{label_base}.csv")

    if not os.path.exists(test_path) or not os.path.exists(label_path):
        print(f"[WARN] Missing data or label for {test_fname}, skipping.")
        continue

    # Load test FCS
    sample = FCMeasurement(ID=label_base, datafile=test_path)
    sample_df = sample.data.copy()

    # Extract same features and scale
    sample_X = sample_df[features].copy()
    sample_X_scaled = scaler.transform(sample_X)

    # Load true labels
    labels_df = pd.read_csv(label_path)
    labels_df["Final_Label"] = labels_df.apply(assign_label, axis=1)
    true_labels = labels_df["Final_Label"].map({"WBC": 1, "Blast": -1}).fillna(0).values

    # Predict using trained model
    preds = model.predict(sample_X_scaled)

    # Evaluate
    print(f"\n[TEST RESULTS] {test_fname}")
    print(confusion_matrix(true_labels, preds, labels=[1, -1]))
    print(classification_report(
        true_labels, preds, target_names=["Normal(WBC)", "Blast/Outlier"]
    ))

print("\n[DEBUG] External testing complete.")

# After all grid search combinations
end_time_all = time.time()
total_elapsed = end_time_all - start_time
hours, rem = divmod(total_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n[DEBUG] Script ended at: {time.ctime(end_time_all)}")
print(f"[DEBUG] Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")


