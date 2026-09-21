import os
import re
import pandas as pd

LABEL_FOLDER = "/storage/mezya.sezen/mphasis/dataset/labels/"

def extract_key(filename):
    m = re.match(r"(BLAST\d+_\d+_P\d+)", filename)
    return m.group(1) if m else None

label_files = [f for f in os.listdir(LABEL_FOLDER) if f.endswith(".csv")]

results = []
for label_file in label_files:
    key = extract_key(label_file)
    if key is None:
        continue
    csv_path = os.path.join(LABEL_FOLDER, label_file)
    df = pd.read_csv(csv_path)
    
    # Filter to WBC singlets only (same as your main pipeline)
    if 'Singlets' in df.columns and 'WBC' in df.columns:
        df = df[(df['Singlets'] == 1) & (df['WBC'] == 1)]
    
    total = len(df)
    blasts = df['Blast'].sum()
    blast_pct = (blasts / total * 100) if total > 0 else 0
    
    results.append({
        'sample': key,
        'total_cells': total,
        'blast_count': int(blasts),
        'blast_pct': round(blast_pct, 3)
    })

df_summary = pd.DataFrame(results).sort_values('blast_pct', ascending=False)
print(df_summary.to_string())
df_summary.to_csv("blast_prevalence_per_sample.csv", index=False)
print(f"\nTotal samples: {len(df_summary)}")
print(f"Median blast %: {df_summary['blast_pct'].median():.3f}%")
print(f"Samples with <1% blasts: {(df_summary['blast_pct'] < 1).sum()}")
print(f"Samples with 1-20% blasts: {((df_summary['blast_pct'] >= 1) & (df_summary['blast_pct'] < 20)).sum()}")
print(f"Samples with >20% blasts: {(df_summary['blast_pct'] >= 20).sum()}")