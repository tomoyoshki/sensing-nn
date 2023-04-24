import json
import csv
from collections import defaultdict

input_json_file = "/home/tkimura4/FoundationSense2/result/finetune_result_mean.json"


def convert_result_to_csv(input_json_file):
    # Read the JSON file
    with open(input_json_file) as f:
        data = json.load(f)

    # Prepare a dictionary to store the extracted information
    csv_data = defaultdict(lambda: defaultdict(list))

    # Extract the required information
    first_row = True
    for key, values in data.items():
        dataset, model, framework, task, label_ratio = key.split("-")
        if first_row == False:
            mean_entry = [
                framework,
                f'{format(round(values["acc"]["mean"], 4), ".4f")} \u00B1 {format(round(values["acc"]["std"], 4), ".4f")}', 
                f'{format(round(values["f1"]["mean"], 4), ".4f")} \u00B1 {format(round(values["f1"]["std"], 4), ".4f")}'
            ]
        else:
             mean_entry = [
                framework,
                f'{format(round(values["acc"]["mean"], 4), ".4f")}', 
                f'{format(round(values["f1"]["mean"], 4), ".4f")}'
             ]
             first_row = True
        # std_entry = [framework, values["acc"]["std"], values["f1"]["std"]]
        csv_data[(dataset, model, task)][label_ratio].append([mean_entry])

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

                # # write std row
                # for i, entry in enumerate(label_ratio_entries):
                #     row.extend(entry[1][1:])
                csvwriter.writerow(row)


if __name__ == "__main__":
    convert_result_to_csv(input_json_file)
