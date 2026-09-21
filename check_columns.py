import os
from FlowCytometryTools import FCMeasurement

def check_fcs_columns(folder_path, required_columns):
    """
    Check if required columns exist in each .fcs file inside a folder.
    If missing, print file name and missing columns.
    """

    # List all FCS files
    fcs_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.fcs')]

    if not fcs_files:
        print("No .fcs files found in the folder.")
        return

    for fcs_file in fcs_files:
        file_path = os.path.join(folder_path, fcs_file)

        try:
            sample = FCMeasurement(ID="sample", datafile=file_path)
            df = sample.data

            existing_cols = set(df.columns)
            required_cols = set(required_columns)

            missing = required_cols - existing_cols

            if missing:
                print(f"Missing columns in {fcs_file}: {sorted(missing)}")
            else:
                print(f"All required columns present in: {fcs_file}")

        except Exception as e:
            print(f"Error reading {fcs_file}: {e}")


# -----------------------
# USAGE EXAMPLE
# -----------------------
folder_path = "/storage/mezya.sezen/mphasis/LAIP29/scaled/"

required_columns = [
    "SSC-A","PerCP-A", 
    "PC7-A", "Horizon V450-A", "Horizon V500-A"
]

check_fcs_columns(folder_path, required_columns)
