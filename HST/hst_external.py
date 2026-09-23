#!/usr/bin/env python3
# External validation of the saved HST fold models on LAIP29 (diagnosis + follow-up samples)

import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    cohen_kappa_score, roc_curve, precision_recall_curve
)


# ==================================================

class HalfSpaceTree:
    """
    Simple offline Half-Space Tree style anomaly detector.
    (Copied verbatim from hst.py so joblib can unpickle saved models.)
    """

    def __init__(self, depth=12, n_trees=50, random_state=None):
        self.depth = int(depth)
        self.n_trees = int(n_trees)
        self.rng = np.random.default_rng(random_state)
        self.trees = []

    def _make_tree(self, mins, maxs, n_features):
        def build(d, mins_loc, maxs_loc):
            if d == 0:
                return {
                    "feat": None, "thresh": None, "left": None, "right": None,
                    "mass": 0.0, "depth": 0
                }
            feat = int(self.rng.integers(0, n_features))
            if maxs_loc[feat] == mins_loc[feat]:
                thresh = float(mins_loc[feat])
            else:
                thresh = float(self.rng.uniform(mins_loc[feat], maxs_loc[feat]))
            mins_l, maxs_l = mins_loc.copy(), maxs_loc.copy()
            maxs_l[feat] = thresh
            mins_r, maxs_r = mins_loc.copy(), maxs_loc.copy()
            mins_r[feat] = thresh
            node = {
                "feat": feat, "thresh": thresh,
                "left": build(d - 1, mins_l, maxs_l),
                "right": build(d - 1, mins_r, maxs_r),
                "mass": 0.0, "depth": 0
            }
            return node
        root = build(self.depth, mins.copy(), maxs.copy())
        self._set_depths(root, depth=0)
        return root

    def _set_depths(self, node, depth):
        if node is None:
            return
        node["depth"] = depth
        self._set_depths(node["left"], depth + 1)
        self._set_depths(node["right"], depth + 1)

    def fit(self, X):
        X = np.asarray(X)
        n_features = X.shape[1]
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        self.trees = [self._make_tree(mins, maxs, n_features) for _ in range(self.n_trees)]
        for x in X:
            for tree in self.trees:
                self._update_mass_single(tree, x)
        return self

    def _update_mass_single(self, node, x):
        node["mass"] += 1.0
        if node["feat"] is None:
            return
        feat = node["feat"]
        if x[feat] <= node["thresh"]:
            self._update_mass_single(node["left"], x)
        else:
            self._update_mass_single(node["right"], x)

    def _score_one_tree(self, node, x, alpha=1e-3, beta=1.0):
        if node is None:
            return 0.0
        m = node["mass"]
        d = node["depth"]
        contrib = 1.0 / (alpha + m) * (1.0 + beta * (1.0 / (1 + d)))
        if node["feat"] is None:
            return contrib
        feat = node["feat"]
        if x[feat] <= node["thresh"]:
            return contrib + self._score_one_tree(node["left"], x, alpha=alpha, beta=beta)
        else:
            return contrib + self._score_one_tree(node["right"], x, alpha=alpha, beta=beta)

    def score_samples(self, X):
        X = np.asarray(X)
        out = np.zeros(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            scores = [self._score_one_tree(t, x) for t in self.trees]
            out[i] = np.mean(scores)
        return out

    def decision_function(self, X):
        return self.score_samples(X)


class HSTWrapper:
    """Wrapper matching hst.py exactly, required for joblib unpickling."""

    def __init__(self, n_trees=50, depth=12, random_state=None):
        self.model = HalfSpaceTree(depth=depth, n_trees=n_trees, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)


# ==================================================
# PATHS 
# ==================================================
HST_MODEL_DIR   = "/storage/mezya.sezen/mphasis/hst_27_2"          
LAIP29_FCS_PATH = "/storage/mezya.sezen/mphasis/LAIP29/scaled/"
LAIP29_LABEL_PATH = "/storage/mezya.sezen/mphasis/LAIP29/labels/"
OUTPUT_PATH = os.path.join(HST_MODEL_DIR, "external_validation_LAIP29")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Same feature set used during HST training (order matters)
features = ["SSC-A", "Horizon V450-A", "Horizon V500-A", "PerCP-A", "PC7-A"]

# The two external samples named in the manuscript
EXTERNAL_SAMPLES = {
    "diagnosis": "LAIP29_8_Dx_P2",
    "follow_up": "LAIP29_9_FU_P3",
}


# ==================================================
def from_fcs(path):
    from FlowCytometryTools import FCMeasurement
    sample = FCMeasurement(ID='X', datafile=path)
    try:
        meta = sample.meta['_channels_']
        instrument_cols = meta['$PnN'].tolist()
    except Exception:
        df = sample.data.copy()
        if 'event_ID' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'event_ID'})
        return df
    df = sample.data.copy()
    if len(df.columns) == len(instrument_cols):
        df.columns = instrument_cols
    if 'event_ID' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'event_ID'})
    return df


def load_external_sample(sample_key):
    """Load one LAIP29 FCS file + its label CSV, merge on event_ID, return X, y."""
    fcs_candidates = glob.glob(os.path.join(LAIP29_FCS_PATH, f"{sample_key}*.fcs"))
    if not fcs_candidates:
        raise FileNotFoundError(f"No FCS file found for {sample_key} in {LAIP29_FCS_PATH}")
    fcs_path = fcs_candidates[0]

    label_candidates = glob.glob(os.path.join(LAIP29_LABEL_PATH, f"{sample_key}*.csv"))
    if not label_candidates:
        raise FileNotFoundError(f"No label file found for {sample_key} in {LAIP29_LABEL_PATH}")
    label_path = label_candidates[0]

    ff = from_fcs(fcs_path)
    labels = pd.read_csv(label_path, index_col=0)
    if 'event_ID' not in ff.columns and labels.index.name == 'event_ID':
        ff = ff.reset_index().rename(columns={'index': 'event_ID'})

    merged = pd.merge(ff, labels, on="event_ID", how="inner")

    missing = [c for c in features + ["Blast"] if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {sample_key}")

    # Apply same singlet/WBC gating used in training if columns are present
    if "Singlets" in merged.columns and "WBC" in merged.columns:
        merged = merged[(merged["Singlets"] == 1) & (merged["WBC"] == 1)]

    X = merged[features].values
    y = merged["Blast"].values.astype(int)
    return X, y, merged


def evaluate_and_plot(y_true, y_scores, tau, sample_name, model_tag):
    y_pred = (y_scores >= tau).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics = {
        "sample": sample_name,
        "model": model_tag,
        "n_events": len(y_true),
        "blast_prevalence": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "auroc": roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else np.nan,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "tau_used": tau,
    }

    # ROC + PR plots
    if len(np.unique(y_true)) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        axes[0].plot(fpr, tpr, lw=2, label=f"AUROC={metrics['auroc']:.3f}")
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"{model_tag} ROC - {sample_name}")
        axes[0].legend()

        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        axes[1].plot(rec, prec, lw=2, label=f"AUPRC={metrics['auprc']:.3f}")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title(f"{model_tag} PR - {sample_name}")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f"{model_tag}_{sample_name}_roc_pr.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xticks([0, 1], ['WBC', 'Blast']); plt.yticks([0, 1], ['WBC', 'Blast'])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{model_tag} Confusion Matrix - {sample_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"{model_tag}_{sample_name}_confusion_matrix.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    return metrics



# ==================================================
fold_model_paths = sorted(glob.glob(os.path.join(HST_MODEL_DIR, "hst_fold*_model.pkl")))
if not fold_model_paths:
    raise FileNotFoundError(f"No saved HST fold models found in {HST_MODEL_DIR}")

print(f"[INFO] Found {len(fold_model_paths)} saved HST fold models.")

all_results = []

for label, sample_key in EXTERNAL_SAMPLES.items():
    print(f"\n[INFO] Loading external sample: {sample_key} ({label})")
    X_ext, y_ext, merged_df = load_external_sample(sample_key)
    print(f"        n_events={len(y_ext)}, blast_prevalence={np.mean(y_ext):.3%}")

    # Score with each fold model, then average scores across folds (ensemble)
    fold_scores = []
    fold_taus = []
    for path in fold_model_paths:
        model_data = joblib.load(path)
        model = model_data["model"]
        tau_fold = model_data["tau_fold"]
        scores = model.decision_function(X_ext)
        fold_scores.append(scores)
        fold_taus.append(tau_fold)

        fold_id = model_data["fold"]
        m = evaluate_and_plot(y_ext, scores, tau_fold, sample_key, f"HST_fold{fold_id}")
        all_results.append(m)

    
    mean_scores = np.mean(np.vstack(fold_scores), axis=0)
    median_tau = float(np.median(fold_taus))
    m_ensemble = evaluate_and_plot(y_ext, mean_scores, median_tau, sample_key, "HST_ensemble")
    all_results.append(m_ensemble)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_PATH, "HST_external_validation_LAIP29.csv"), index=False)

print("\n=== HST EXTERNAL VALIDATION COMPLETE ===")
print(results_df[["sample", "model", "n_events", "blast_prevalence",
                   "accuracy", "precision", "recall", "f1", "auroc", "auprc", "kappa"]])
print(f"\n[OUTPUT] {OUTPUT_PATH}")
