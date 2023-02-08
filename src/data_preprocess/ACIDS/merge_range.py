import os
import json


input_path = "/home/kara4/data/ACIDS/partition_ranges_one_sec_snr"
output_file = "/home/sl29/data/ACIDS/mat_label_range.json"

merged_ranges = {}
for file in [
    "background_ranges_for_deepsense_labeling_train.json",
    "background_ranges_for_deepsense_labeling_val.json",
    "background_ranges_for_deepsense_labeling_test.json",
]:
    with open(os.path.join(input_path, file), "r") as f:
        input_ranges = json.load(f)
        for mat_file in input_ranges:
            if mat_file not in merged_ranges:
                merged_ranges[mat_file] = input_ranges[mat_file]

        print(file, len(input_ranges))
        print(set(input_ranges.keys()))
        print("=" * 100)

with open(output_file, "w") as f:
    f.write(json.dumps(merged_ranges, indent=4))
