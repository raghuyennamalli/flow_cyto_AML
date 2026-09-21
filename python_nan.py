import pandas as pd
import os
import numpy as np
from FlowCytometryTools import FCMeasurement

# Directories
healthy_dir = "/storage/mezya.sezen/mphasis/dataset/after_scaling/"
total_dir = "/storage/mezya.sezen/mphasis/dataset/after_scaling/ "

features = ["SSC-A", "CD45 KO", "CD34 Cy55", "CD117"]

def check_missing_values(directory, features, output_csv):
    results = []
    
    for file in os.listdir(directory):
        if file.endswith(".fcs"):
            path = os.path.join(directory, file)
            sample = FCMeasurement(ID=file, datafile=path).data
            
            # Count missing values
            missing_counts = sample[features].isna().sum()
            total_counts = len(sample)
            missing_percentage = (missing_counts / total_counts) * 100
            
            result = pd.DataFrame({
                "Feature": features,
                "Missing_Count": missing_counts.values,
                "Total_Events": total_counts,
                "Missing_%": missing_percentage.values
            })
            result["File"] = file
            results.append(result)
    
    final_df = pd.concat(results, ignore_index=True)
    
    # Summary across all files
    summary = final_df.groupby("Feature")[["Missing_Count", "Missing_%"]].mean()
    print("\nAverage missing counts and percentages across all files:")
    print(summary)
    
    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved missing value report ? {output_csv}")

# Run for both datasets
check_missing_values(healthy_dir, features, "/home/mezya.sezen/missing_value_summary_HEALTHY.csv")
check_missing_values(total_dir, features, "/home/mezya.sezen/missing_value_summary_ALL.csv")
