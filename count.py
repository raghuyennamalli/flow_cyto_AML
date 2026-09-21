import os
import re
from collections import defaultdict

def check_fcs_files(folder_path, output_file="results_v2.txt"):
    # Regex to match files like BLAST110_13_P2.fcs
    pattern = re.compile(r"(BLAST110_(\d+))_P(\d)\.fcs")

    # Dictionaries to store counts and presence
    counts = defaultdict(int)
    presence = defaultdict(set)

    # Scan folder
    for file in os.listdir(folder_path):
        match = pattern.match(file)
        if match:
            base, sample_id, pnum = match.groups()
            counts[f"P{pnum}"] += 1
            presence[base].add(f"P{pnum}")

    # Collect output lines
    output_lines = []

    # Counts table
    output_lines.append("Counts Summary\n")
    output_lines.append("+--------+-------+\n")
    output_lines.append("| Type   | Count |\n")
    output_lines.append("+--------+-------+\n")
    for i in range(1, 5):
        output_lines.append("| {:<6} | {:>5} |\n".format(f"P{i}", counts[f"P{i}"]))
    output_lines.append("+--------+-------+\n\n")

    # Missing files detailed
    missing = defaultdict(list)
    for base, ps in presence.items():
        # Extract numeric sample ID from base (BLAST110_13 → 13)
        sample_id = base.split("_")[1]
        for i in range(1, 5):
            if f"P{i}" not in ps:
                missing[f"P{i}"].append(sample_id)

    # Print grouped missing summary
    output_lines.append("Missing Files (Grouped by P type)\n")
    for i in range(1, 5):
        if missing[f"P{i}"]:
            output_lines.append(f"P{i} missing: {', '.join(missing[f'P{i}'])}\n")
        else:
            output_lines.append(f"P{i} missing: none\n")

    # Save to text file
    result_path = os.path.join("/storage/mezya.sezen/mphasis/dataset", output_file)
    with open(result_path, "w") as f:
        f.writelines(output_lines)

    print(f"Results saved to {result_path}")

# Example usage
folder = "/storage/mezya.sezen/mphasis/dataset/BLAST110/"
check_fcs_files(folder)