import os
import time
import numpy as np
import pandas as pd
import joblib

from FlowCytometryTools import FCMeasurement
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from sklearn.metrics._scorer import _BaseScorer


# ==============================
# PATHS & GLOBALS
# ==============================
FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels"
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/lof_results/"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "CV"), exist_ok=True)

features = ["SSC-A", "Horizon V450-A", "Horizon V500-A", "PerCP-A", "PC7-A"]


# ==============================
# DATA PREP
# ==============================
def prepare_dataset():
    pkl_path = os.path.join(OUTPUT_PATH, "BLAST110_5K.pkl")

    if os.path.exists(pkl_path):
        print(f"[INFO] Loading cached dataset: {pkl_path}")
        return pd.read_pickle(pkl_path)

    print("[INFO] Building dataset from raw FCS files...")
    dfs = []

    for root, dirs, files in os.walk(FCS_PATH):
        for file in files:
            if not file.endswith(".fcs"):
                continue

            name = file.replace(".fcs", "")
            parts = name.split("_")
            patient_id = "_".join(parts[:2])
            sample_id = parts[2]

            fcs_file = os.path.join(root, file)
            label_file = os.path.join(LABEL_PATH, f"{patient_id}_{sample_id}.csv")

            if not os.path.exists(label_file):
                print(f"[WARNING] Missing label for {patient_id}_{sample_id}")
                continue

            try:
                df_fcs = FCMeasurement(ID="sample", datafile=fcs_file).data.copy()
                df_labels = pd.read_csv(label_file, index_col=0)

                df = pd.merge(df_fcs, df_labels, on="event_ID", how="inner")
                df["patient_id"] = patient_id
                df["sample_id"] = sample_id

                if len(df) >= 5000:
                    dfs.append(df.sample(5000, random_state=42))
                else:
                    print(f"[INFO] Sample {sample_id} < 5000 events, skipping.")

            except Exception as e:
                print(f"[ERROR] Failed on {file}: {e}")

    if not dfs:
        raise RuntimeError("No data processed. Check paths.")

    data = pd.concat(dfs).reset_index(drop=True)
    data.to_pickle(pkl_path)
    print(f"[INFO] Saved dataset to: {pkl_path}")

    return data


# ==============================
# CUSTOM LOF SCORER
# ==============================
from sklearn.metrics import make_scorer

# Custom LOF scoring function
# Only estimator and X are needed for LOF; y=None for GridSearchCV compatibility
def lof_norm_score(estimator, X, y=None):
    try:
        scores = estimator.decision_function(X)
        return np.mean(scores)  # maximize 'normality'
    except Exception:
        return 0

lof_scorer = lof_norm_score





# ==============================
# THRESHOLDING FUNCTION
# ==============================
def find_optimal_threshold(y_true, y_scores, method="clinical_recall"):
    # LOF: higher = normal, lower = blast
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        gmeans = np.sqrt(tpr * (1 - fpr))
        thr = thresholds[np.argmax(gmeans)]

    elif method == "f1":
        p, r, thr = precision_recall_curve(y_true, y_scores)
        f = (2 * p * r) / (p + r + 1e-12)
        best = np.argmax(f)
        thr = thr[best] if best < len(thr) else thr[-1]

    elif method == "clinical_recall":
        p, r, thr = precision_recall_curve(y_true, y_scores)
        thr = np.append(np.min(thr) - 1, thr)

        mask = p >= 0.9
        if np.sum(mask) > 0:
            best = np.argmax(r[mask])
            thr = thr[mask][best]
        else:
            f = (2 * p * r) / (p + r + 1e-12)
            best = np.argmax(f)
            thr = thr[best] if best < len(thr) else thr[-1]

    else:
        raise ValueError("Invalid thresholding method")

    preds = (y_scores < thr).astype(int)

    return thr, {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


# ==============================
# MAIN LOF PIPELINE
# ==============================
def run_lof_pipeline(data):
    X = data[features].values
    y = data["Blast"].values
    groups = data["patient_id"].values
    samples = data["sample_id"].values

    print(f"[INFO] Data = {X.shape}, blast rate = {y.mean():.4f}")

    outer_cv = GroupKFold(n_splits=3)
    inner_cv = GroupKFold(n_splits=3)

    param_grid = {
        "n_neighbors": [20, 50, 100],
        "contamination": [0.01, 0.02, 0.03],
    }

    best_global_f1 = -1
    best_global_model = None

    all_results = []
    summary = []

    t0 = time.time()

    for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups)):
        print(f"\n=== OUTER FOLD {fold + 1} ===")

        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        gtr = groups[tr]

        Xtr_norm = Xtr[ytr == 0]

        grid = GridSearchCV(
              LocalOutlierFactor(novelty=True),
              param_grid,
              cv=inner_cv.split(Xtr_norm, groups=gtr[ytr == 0]),
              scoring=lof_scorer,
              n_jobs=4,
              refit=True,
              verbose=1,
              )


        t_inner = time.time()
        grid.fit(Xtr_norm)
        print(f"[INNER] Best params: {grid.best_params_}")
        print(f"[INNER] Time: {time.time() - t_inner:.1f}s")

        model = grid.best_estimator_
        scores = model.decision_function(Xte)

        thr, m = find_optimal_threshold(yte, scores, "clinical_recall")
        print(f"[FOLD] Threshold={thr:.4f}, Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1']:.3f}")

        if m["f1"] > best_global_f1:
            best_global_f1 = m["f1"]
            best_global_model = model
            print(f"[INFO] Updated global best model (F1={best_global_f1:.3f})")

        summary.append({
            "fold": fold,
            "best_params": grid.best_params_,
            "threshold": thr,
            **m
        })

        # Per sample logs
        for sid in np.unique(samples[te]):
            mask = samples[te] == sid
            y_sub = yte[mask]
            s_sub = scores[mask]
            preds = (s_sub < thr).astype(int)

            all_results.append({
                "fold": fold,
                "sample_id": sid,
                "gt_count": y_sub.sum(),
                "gt_perc": y_sub.mean(),
                "pred_count": preds.sum(),
                "pred_perc": preds.mean(),
                "precision": precision_score(y_sub, preds, zero_division=0),
                "recall": recall_score(y_sub, preds, zero_division=0),
                "f1": f1_score(y_sub, preds, zero_division=0),
            })

    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_PATH, "CV/threshold_summary_LOF.csv"), index=False)
    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_PATH, "CV/all_samples_LOF_results.csv"), index=False)

    print(f"\n[DONE] Runtime: {time.time() - t0:.1f}s")

    if best_global_model:
        joblib.dump(best_global_model, os.path.join(OUTPUT_PATH, "best_LOF_model.pkl"))
        print("[INFO] Saved best LOF model.")


# ==============================
# MAIN
# ==============================
def main():
    data = prepare_dataset()
    run_lof_pipeline(data)


if __name__ == "__main__":
    main()
