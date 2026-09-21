import os
from FlowCytometryTools import FCMeasurement

# === CONFIG ===
fcs_folder =  "/storage/mezya.sezen/mphasis/LAIP29/scaled/"
csv_folder = "/storage/mezya.sezen/mphasis/LAIP29/labels" 
required_channels = ["SSC-A", "Horizon V500-A", "PerCP-A", "PC7-A"]

## === COLLECT FILES ===
fcs_files = [f for f in os.listdir(fcs_folder) if f.endswith(".fcs")]
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

print(f"Total FCS files: {len(fcs_files)}")
print(f"Total CSV files: {len(csv_files)}")

# === HELPERS ===
def get_id_from_fcs(name):
    parts = name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return os.path.splitext(name)[0]

missing_csv_files = []
missing_columns_files = {}

# === MAIN CHECK ===
for fcs_file in fcs_files:
    fcs_id = get_id_from_fcs(fcs_file)
    fcs_path = os.path.join(fcs_folder, fcs_file)
    csv_name = f"{fcs_id}.csv"
    csv_path = os.path.join(csv_folder, csv_name)

    print(f"\nChecking: {fcs_file}")

    # --- Check CSV presence ---
    if not os.path.exists(csv_path):
        print(f"  ✗ Missing CSV file: {csv_name}")
        missing_csv_files.append(fcs_file)
        continue

    # --- Check FCS columns ---
    try:
        sample = FCMeasurement(ID=fcs_file, datafile=fcs_path)
        columns = list(sample.data.columns)
    except Exception as e:
        print(f"  ✗ Error reading FCS file: {e}")
        continue

    missing_cols = [c for c in required_channels if c not in columns]
    if missing_cols:
        print(f"  ✗ Missing columns: {missing_cols}")
        missing_columns_files[fcs_file] = missing_cols
    else:
        print("  ✓ All required columns present")

# === SUMMARY ===
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

if missing_csv_files:
    print("\nFiles missing corresponding CSV:")
    for f in missing_csv_files:
        print(" -", f)
else:
    print("\nAll FCS files have corresponding CSV files.")

if missing_columns_files:
    print("\nFiles missing required columns:")
    for f, cols in missing_columns_files.items():
        print(f" - {f}: Missing {cols}")
else:
    print("\nAll FCS files contain required columns.")

print("\nCheck complete.")

# === SELECT ONE .fcs FILE ===
fcs_files = [f for f in os.listdir(fcs_folder) if f.endswith(".fcs")]

if not fcs_files:
    print("No FCS files found!")
else:
    fcs_file = fcs_files[0]      # take the first file
    fcs_path = os.path.join(fcs_folder, fcs_file)

    print(f"\nReading FCS file: {fcs_file}")

    # Load
    sample = FCMeasurement(ID=fcs_file, datafile=fcs_path)

    # Extract data column names (parameter names)
    cols = list(sample.data.columns)

    print("\n==============================")
    print("COLUMN NAMES (from data):")
    print("==============================")
    for c in cols:
        print(c)
    sample = FCMeasurement(ID=fcs_file, datafile=fcs_path)
    columns = list(sample.data.columns)
    meta = sample.meta
    mapping = {}
    for i, col in enumerate(columns, start=1):
      marker = meta.get(f"$P{i}S", "NO_MARKER")
      mapping[col] = marker
    print("\nMAPPED: COLUMN NAME  -->  MARKER NAME")
    print("="*50)
    for col, marker in mapping.items():
      print(f"{col:20s} --> {marker}")
