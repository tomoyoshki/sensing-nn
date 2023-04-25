import json
import csv
from collections import defaultdict

# input_json_file = "/home/tkimura4/FoundationSense2/result/finetune_result_mean.json"
# input_json_file = "/home/sl29/FoundationSense/result/finetune_result_mean.json"
input_json_file = "./knn_result.json"


def convert_result_to_csv(input_json_file):
    # Read the JSON file
    with open(input_json_file) as f:
        data = json.load(f)

    # Prepare a dictionary to store the extracted information
    # csv_data = defaultdict(lambda: defaultdict(list))
    csv_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Extract the required information
    for key, values in data.items():
        dataset, model, framework, task, label_ratio = key.split("-")
        if not label_ratio == "1.0":
            mean_entry = [
                framework,
                f'{format(round(values["acc"]["mean"], 4), ".4f")} \u00B1 {format(round(values["acc"]["std"], 4), ".4f")}',
                f'{format(round(values["f1"]["mean"], 4), ".4f")} \u00B1 {format(round(values["f1"]["std"], 4), ".4f")}',
            ]
        else:
            mean_entry = [
                framework,
                f'{format(round(values["acc"]["mean"], 4), ".4f")}',
                f'{format(round(values["f1"]["mean"], 4), ".4f")}',
            ]
            first_row = False
        print(mean_entry)
        # std_entry = [framework, values["acc"]["std"], values["f1"]["std"]]
        csv_data[(dataset, model, task)][framework][label_ratio] = mean_entry

    # Write the information to CSV files
    for (dataset, model, task), frameworks_data in csv_data.items():
        file_name = f"{dataset}_{model}_{task}.csv"

        # Specify the label ratios
        sorted_label_ratios = ["1.0", "0.1", "0.01"]
        if "knn" in input_json_file:
            sorted_label_ratios = ["1.0"]

        # Write to the CSV file
        with open(file_name, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header
            header = ["Framework"]
            for label_ratio in sorted_label_ratios:
                header.extend([f"ACC_{label_ratio}_mean", f"F1_{label_ratio}_mean"])
            csvwriter.writerow(header)

            # # Write data
            # Write data
            frameworks = [
                "SimCLR",
                "MoCo",
                "CMC",
                "CMCV2",
                "MAE",
                "Cosmo",
                "Cocoa",
                "MTSS",
                "TS2Vec",
                "GMC",
                "TNC",
                "TSTCC",
            ]
            for framework in frameworks:
                # print(framework)
                label_ratios_data = frameworks_data[framework]
                row = [framework]
                for label_ratio in sorted_label_ratios:
                    if label_ratio in label_ratios_data:
                        entry = label_ratios_data[label_ratio]
                        row.extend(entry[1:])
                    else:
                        row.extend(["", ""])
                # print(row)
                csvwriter.writerow(row)


if __name__ == "__main__":
    convert_result_to_csv(input_json_file)
