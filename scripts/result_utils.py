import json
import csv
from collections import defaultdict

input_json_file = "./knn_result_mean.json"


def convert_result_to_csv(input_json_file):
    # Read the JSON file
    with open(input_json_file) as f:
        data = json.load(f)

    # Prepare a dictionary to store the extracted information
    csv_data = defaultdict(lambda: defaultdict(list))

    # Extract the required information
    for key, values in data.items():
        dataset, model, framework, task, label_ratio = key.split("-")
        mean_entry = [framework, values["acc"]["mean"], values["f1"]["mean"]]
        std_entry = [framework, values["acc"]["std"], values["f1"]["std"]]
        csv_data[(dataset, model, task)][label_ratio].append([mean_entry, std_entry])

    # Write the information to CSV files
    for (dataset, model, task), label_ratios_data in csv_data.items():
        file_name = f"{dataset}_{model}_{task}.csv"

        # Specify the label ratios
        sorted_label_ratios = ["1.0", "0.1", "0.01"]

        # Write to the CSV file
        with open(file_name, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header
            header = ["Framework"]
            for label_ratio in sorted_label_ratios:
                header.extend([f"ACC_{label_ratio}_mean", f"F1_{label_ratio}_mean"])

            for label_ratio in sorted_label_ratios:
                header.extend([f"ACC_{label_ratio}_std", f"F1_{label_ratio}_std"])
            csvwriter.writerow(header)

            # Write data
            for label_ratio_entries in zip(*[label_ratios_data[label_ratio] for label_ratio in sorted_label_ratios]):
                row = []

                # write mean row
                for i, entry in enumerate(label_ratio_entries):
                    if i == 0:
                        row.extend(entry[0])
                    else:
                        row.extend(entry[0][1:])

                # write std row
                for i, entry in enumerate(label_ratio_entries):
                    row.extend(entry[1][1:])
                csvwriter.writerow(row)


if __name__ == "__main__":
    convert_result_to_csv(input_json_file)
