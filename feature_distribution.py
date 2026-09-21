import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from FlowCytometryTools import FCMeasurement
import seaborn as sns


FCS_PATH = "/storage/mezya.sezen/mphasis/dataset/final_scaled/" 
LABEL_PATH = "/storage/mezya.sezen/mphasis/dataset/labels" 
OUTPUT_PATH = "/storage/mezya.sezen/mphasis/Features_plot"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def from_fcs(path):
    sample = FCMeasurement(ID="sample", datafile=path)
    return sample.data.copy()

SCATTER_FEATURES = ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H']
FLUORESCENCE_FEATURES = ['PerCP-A', 'PC7-A', 'APC-H7-A', 'Horizon V450-A', 'Horizon V500-A']
features = SCATTER_FEATURES + FLUORESCENCE_FEATURES

print("FEATURE COLUMNS TO USE (COMMON ACROSS ALL FILES, EXCLUDING METADATA):")
print(features)

# -----------------------------STEP 1 — GENERATE PKL FILES -----------------------------
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

            ff = ff[features + ["Blast", "Singlets", "event_ID"]]

            ff = ff[(ff["Singlets"] == 1) ]
            ff = ff.drop(columns=["Singlets", "event_ID"])

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
df = pd.read_pickle(pkl_5k)


output_dir1 = os.path.join(OUTPUT_PATH, "kde_plots")
os.makedirs(output_dir1, exist_ok=True)

output_dir2 = os.path.join(OUTPUT_PATH, "violin_plots")
os.makedirs(output_dir2, exist_ok=True)

# ==================== TWO-SEPARATE-SCALE VERSION ====================

# 1. Scatter features (FSC/SSC) - ONE scale
SCATTER_FEATURES = ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H']
scatter_min = df[SCATTER_FEATURES].min().min()
scatter_max = df[SCATTER_FEATURES].max().max()

# 2. Fluorescence features - ANOTHER scale  
FLUORESCENCE_FEATURES = ['PerCP-A', 'PC7-A', 'APC-H7-A', 'Horizon V450-A', 'Horizon V500-A']
fluor_min = df[FLUORESCENCE_FEATURES].min().min()
fluor_max = df[FLUORESCENCE_FEATURES].max().max()

print(f"[SCALES]")
print(f"  FSC/SSC: [{scatter_min:.1f}, {scatter_max:.1f}]")
print(f"  Fluorescence: [{fluor_min:.1f}, {fluor_max:.1f}]")

# ==================== KDE PLOTS ====================
output_dir_kde = os.path.join(OUTPUT_PATH, "kde_plots_two_scales")
os.makedirs(output_dir_kde, exist_ok=True)

for feat in features:
    plt.figure(figsize=(6,4))
    blast_vals = df[df["Blast"]==1][feat].dropna()
    non_vals = df[df["Blast"]==0][feat].dropna()

    if len(blast_vals) > 1 and len(non_vals) > 1:
        kde_b = gaussian_kde(blast_vals)
        kde_n = gaussian_kde(non_vals)
        
        # CHOOSE SCALE based on feature type
        if feat in SCATTER_FEATURES:
            x_min, x_max = scatter_min, scatter_max
        else:  # Fluorescence
            x_min, x_max = fluor_min, fluor_max
            
        xs = np.linspace(x_min, x_max, 400)

        plt.fill_between(xs, kde_b(xs), color='blue', alpha=0.5)
        plt.fill_between(xs, kde_n(xs), color='red', alpha=0.5)
        plt.plot(xs, kde_b(xs), color='blue', linestyle='--', linewidth=1.5)
        plt.plot(xs, kde_n(xs), color='red', linestyle='--', linewidth=1.5)
        plt.xlim(x_min, x_max)  # CORRECT SCALE

    plt.title(f"KDE — {feat}")
    plt.xlabel(feat)
    plt.ylabel("Density")
    plt.legend(["Blast", "Non-Blast"])
    plt.savefig(os.path.join(output_dir_kde, f"{feat}_kde.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==================== VIOLIN PLOTS ====================
output_dir_violin = os.path.join(OUTPUT_PATH, "violin_plots_two_scales")
os.makedirs(output_dir_violin, exist_ok=True)

for feat in features:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='Blast', y=feat, data=df, inner='quartile', palette={0:'red',1:'blue'})
    
    # SAME SCALE LOGIC
    if feat in SCATTER_FEATURES:
        plt.ylim(scatter_min, scatter_max)
    else:
        plt.ylim(fluor_min, fluor_max)
        
    plt.title(f"Violin — {feat}")
    plt.xlabel("Blast")
    plt.ylabel(feat)
    plt.savefig(os.path.join(output_dir_violin, f"{feat}_violin.png"), dpi=300, bbox_inches='tight')
    plt.close()
