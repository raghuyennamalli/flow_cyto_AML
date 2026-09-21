import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, dendrogram

from fcsparser import parse   # make sure fcsparser is installed


# -----------------------------
# CONFIGURATION
# -----------------------------
FEATURES = ["SSC-A", "CD45 KO", "CD34 Cy55", "CD117"]

N_BASE_CLUSTERS = 50
LINKAGE_METHOD = "ward"

FCS_FOLDER = "/storage/mezya.sezen/mphasis/dataset/final_scaled/"   # <<< CHANGE THIS
OUTPUT_PREFIX = "centroid_hierarchical"


# -----------------------------
# UTILS
# -----------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# LOAD FCS FILES
# -----------------------------
def load_all_fcs(folder_path):
    start = time.time()
    print(f"[{now()}] Starting to load FCS files...")

    all_files = glob.glob(os.path.join(folder_path, "*.fcs"))
    dfs = []

    for file in all_files:
        print(f"[{now()}] Reading file: {file}")
        try:
            meta, data = parse(file, reformat_meta=True)
            sample_id = os.path.basename(file).replace(".fcs", "")
            data["sample_id"] = sample_id
            dfs.append(data)
        except Exception as e:
            print(f"[{now()}] Error reading {file}: {e}")

    if len(dfs) == 0:
        raise RuntimeError(f"No FCS files found in {folder_path}")

    df = pd.concat(dfs, ignore_index=True)

    # sanity check
    required_cols = FEATURES 
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in FCS data: {missing}")

    print(f"[{now()}] Combined {len(all_files)} files with {len(df)} events.")
    print(f"[{now()}] Time taken: {round((time.time() - start)/60, 2)} min")

    return df


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    # --- load ---
    df = load_all_fcs(FCS_FOLDER)

    # --- clean ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)

    print(f"[{now()}] Remaining events after cleaning: {len(df)}")

    # --- scale ---
    X = df[FEATURES].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- base clustering (cells) ---
    kmeans = MiniBatchKMeans(
        n_clusters=N_BASE_CLUSTERS,
        random_state=42,
        batch_size=4096,
        n_init=10
    )

    df["cell_cluster"] = kmeans.fit_predict(X_scaled)

    # --- centroid computation ---
    cluster_summary = (
    df.groupby("cell_cluster")
      .agg(
          n_cells=("cell_cluster", "size"),
          **{f"{m}_median": (m, "median") for m in FEATURES}
      )
      .reset_index()
    )

    print(f"[{now()}] Computed {len(cluster_summary)} base clusters")

    # --- hierarchical clustering on centroids ---
    centroid_matrix = cluster_summary[[f"{m}_median" for m in FEATURES]].values

    Z = linkage(
        centroid_matrix,
        method=LINKAGE_METHOD,
        metric="euclidean"
    )

    # --- dendrogram ---
    plt.figure(figsize=(10, 6))
    dendrogram(
        Z,
        labels=cluster_summary["cell_cluster"].astype(str).values,
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title("Hierarchical clustering of cell populations (centroids)")
    plt.xlabel("Base clusters")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PREFIX}_dendrogram.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # --- save summary ---
    cluster_summary.to_csv(
        f"{OUTPUT_PREFIX}_cluster_summary.csv",
        index=False
    )

    print(f"[{now()}] Analysis complete.")
