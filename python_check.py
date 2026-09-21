from fcsparser import parse

fcs_file = "/storage/mezya.sezen/mphasis/dataset/BLAST110/BLAST110_1_P1.fcs"
meta, data = parse(fcs_file, reformat_meta=True)

print("=== Metadata keys available ===")
for key in meta.keys():
    print(key)

print("\n=== First few columns in data ===")
print(list(data.columns)[:20])
