import os
import time
import getpass

import numpy as np
import pandas as pd


def process_one_subject(input_file_name, input_path, output_path):
    """Extract clean labels for one subject.

    Args:
        input_file (_type_): _description_
        input_path (_type_): _description_
        output_path (_type_): _description_
    """
    user_id = input_file_name.split(".")[0]
    output_file = os.path.join(output_path, f"{user_id}.csv")

    df = pd.read_csv(filepath_or_buffer=os.path.join(input_path, input_file_name), delimiter=",", header=0, index_col=0)
    df = df.filter(regex="timestamp|label:", axis=1)

    # replace nan with 0
    df = df.fillna(0)

    # drop location labels: [LOC_main_workplace, LOC_mainhome, LOC_beach]
    df = df[df.columns.drop(list(df.filter(regex="LOC")))]

    # extract the label columns
    # label_cols = [col for col in df.columns if "label:" in col]
    # print(label_cols)

    # only extracted interested classes
    df = df[
        [
            "label:LYING_DOWN",
            "label:SITTING",
            # "label:FIX_walking",
            # "label:FIX_running",
            # "label:BICYCLING",
            "label:OR_standing",
        ]
    ]

    # check num classses, and remove samples with no class
    df["label_count"] = df.sum(axis=1, numeric_only=True)
    df = df[df.label_count == 1]

    # save the extracted clean labels
    df = df.drop(labels=["label_count"], axis=1)

    # save df to output file
    df.to_csv(output_file)


if __name__ == "__main__":
    username = getpass.getuser()
    input_path = f"/home/{username}/data/ExtraSensory/primary_features_and_labels"
    output_path = f"/home/{username}/data/ExtraSensory/clean_labels"

    for fn in os.listdir(input_path):
        if fn.endswith(".csv"):
            process_one_subject(fn, input_path, output_path)
