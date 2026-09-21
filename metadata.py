from FlowCytometryTools import FCMeasurement

# ==============================================================
# CHANGE THIS to your FCS file path
# ==============================================================
FCS_FILE = "/storage/mezya.sezen/mphasis/dataset/final_scaled/BLAST110_100_P1_Cleaned_transformed_scaled.fcs"      # <-- update this

print("\n==============================")
print("LOADING FCS FILE")
print("==============================")

fcs = FCMeasurement(ID="sample", datafile=FCS_FILE)
meta = fcs.meta
df = fcs.data

print("\n==============================")
print("COLUMN NAMES AS READ BY THE ENVIRONMENT")
print("==============================")
print(df.columns.tolist())

print("\n==============================")
print("FULL METADATA CONTENTS ($PnN, $PnS, etc.)")
print("==============================")
for k, v in meta.items():
    print(f"{k}: {v}")

print("\n==============================")
print("PARAMETER INDEX ? CHANNEL NAME ($PnN) AND MARKER ($PnS)")
print("==============================")

for i in range(1, 50):  # check up to 50 channels
    pN = meta.get(f"$P{i}N")
    pS = meta.get(f"$P{i}S")
    if pN or pS:
        print(f"P{i}:  $P{i}N = {pN}    |    $P{i}S = {pS}")

