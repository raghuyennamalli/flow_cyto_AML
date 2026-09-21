import os
import random
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from FlowCytometryTools import FCMeasurement
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
import time


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
# Step 1: Collect data from all files
# -------------------------------
all_data = []
common_features = None  # will store intersection of common markers

for fname in os.listdir(fcs_folder):
    if fname.endswith("_cleaned_doubletRemoval_transformed_scaled.fcs"):
        fpath = os.path.join(fcs_folder, fname)
        base = os.path.splitext(fname)[0]
        label_base = base.replace("_cleaned_doubletRemoval_transformed_scaled", "")
        label_path = os.path.join(labels_folder, f"{label_base}.csv")

        if not os.path.exists(label_path):
            print(f"[DEBUG] Missing label file for {fname}, skipping...")
            continue

        print(f"[DEBUG] Processing: {fname}")

        # Load FCS
        sample = FCMeasurement(ID=label_base, datafile=fpath)
        features_df = sample.data.copy()

        # Initialize or update common feature set
        if common_features is None:
            common_features = set(features_df.columns)
        else:
            common_features &= set(features_df.columns)

        # Load and label
        labels_df = pd.read_csv(label_path)
        labels_df["Final_Label"] = labels_df.apply(assign_label, axis=1)

        merged = features_df.merge(
            labels_df[["event_ID", "Final_Label"]],
            on="event_ID",
            how="inner"
        )
        all_data.append(merged)

# -------------------------------
# Step 2: Restrict to common columns only
# -------------------------------
if len(all_data) == 0:
    raise RuntimeError("No data was collected. Check file paths or filenames.")

# Add scatter channels explicitly (if not already present)
essential_channels = {"FSC-A", "SSC-A"}
common_features |= essential_channels

print(f"[DEBUG] Common markers across all tubes: {sorted(common_features)}")

# Keep only these features for consistency
for i in range(len(all_data)):
    all_data[i] = all_data[i].loc[:, list(common_features) + ["event_ID", "Final_Label"]]

# Concatenate
data = pd.concat(all_data, ignore_index=True)
print(f"[DEBUG] Final dataset shape: {data.shape}")
print(f"[DEBUG] Label distribution:\n{data['Final_Label'].value_counts()}")

# -------------------------------
# Step 3: Prepare features and labels
# -------------------------------
# Use only backbone/common markers for training
features = ["SSC-A", "Horizon V500-A", "PerCP-A", "PC7-A"]
print(f"[DEBUG] Using features: {features}")

X = data[features].copy()
y = data["Final_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split separately for WBC and Blast to preserve balance
X_wbc = X_scaled[y == "WBC"]
X_blast = X_scaled[y == "Blast"]

# --- Split into 60-20-20 ---
# For WBC (normal) data
X_train_wbc, X_temp_wbc = train_test_split(X_wbc, test_size=0.4, random_state=42)
X_val_wbc, X_test_wbc = train_test_split(X_temp_wbc, test_size=0.5, random_state=42)

# For Blast (abnormal) data
X_train_blast, X_temp_blast = train_test_split(X_blast, test_size=0.4, random_state=42)
X_val_blast, X_test_blast = train_test_split(X_temp_blast, test_size=0.5, random_state=42)

# Training uses only WBC (normal)
X_train = np.nan_to_num(X_train_wbc, nan=0.0, posinf=0.0, neginf=0.0)

# Validation and testing sets combine WBC + Blast
X_val = np.vstack([X_val_wbc, X_val_blast])
y_val = np.array([1] * len(X_val_wbc) + [-1] * len(X_val_blast))

X_test = np.vstack([X_test_wbc, X_test_blast])
y_test = np.array([1] * len(X_test_wbc) + [-1] * len(X_test_blast))

print(f"[DEBUG] Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
print(f"[DEBUG] Validation label counts: {{1: {(y_val == 1).sum()}, -1: {(y_val == -1).sum()}}}")
print(f"[DEBUG] Test label counts: {{1: {(y_test == 1).sum()}, -1: {(y_test == -1).sum()}}}")

# -------------------------------
# Step 4: Parameter Grid Search (Approximate SVM)
# -------------------------------


param_grid = {
    "sgdoneclasssvm__nu": [0.03, 0.05, 0.1],
    "sgdoneclasssvm__learning_rate": ["optimal", "constant"],
    "sgdoneclasssvm__max_iter": [1000, 5000],
    "nystroem__n_components": [500, 1000, 2000],
    "nystroem__gamma": [1.0, 0.1, 0.01]
}


print("\n=== Starting Approximate One-Class SVM (SGD + Nystroem) ===\n")

for params in ParameterGrid(param_grid):
    print(f"[DEBUG] Testing Parameters: {params}")
    start_time = time.time()
    try:
        pipeline = make_pipeline(
            Nystroem(kernel='rbf'),
            SGDOneClassSVM()
        )
        pipeline.set_params(**params)
        pipeline.fit(X_train)

        # Evaluate on validation set
        val_preds = pipeline.predict(X_val)
        print("\n[VALIDATION RESULTS]")
        print(confusion_matrix(y_val, val_preds, labels=[1, -1]))
        print(classification_report(
            y_val, val_preds, target_names=["Normal(WBC)", "Blast/Outlier"]
        ))

        # Evaluate on test set
        test_preds = pipeline.predict(X_test)
        print("\n[TEST RESULTS]")
        print(confusion_matrix(y_test, test_preds, labels=[1, -1]))
        print(classification_report(
            y_test, test_preds, target_names=["Normal(WBC)", "Blast/Outlier"]
        ))

    except Exception as e:
        print(f"[ERROR] Failed for parameters {params}: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"[DEBUG] Runtime for this config: {int(minutes)}m {int(seconds)}s")
    print("-" * 60)

# After all grid search combinations
end_time_all = time.time()
total_elapsed = end_time_all - start_time_all
hours, rem = divmod(total_elapsed, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n[DEBUG] Script ended at: {time.ctime(end_time_all)}")
print(f"[DEBUG] Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")

