import os
import os
import json
import time

import numpy as np

from test import test
from params.base_params import parse_base_args
from params.params_util import set_auto_params
from finetune_script import check_execution_flags
from collections import defaultdict


def update_finetune_result(result, result_file):
    """
    dataset --> model --> task --> learn_framework --> label_ratio -- > {acc, f1}
    """
    if os.path.exists(result_file):
        complete_result = json.load(open(result_file))
    else:
        complete_result = {}

    for config in result:
        complete_result[config] = result[config]

    with open(result_file, "w") as f:
        f.write(json.dumps(complete_result, indent=4))


def test_loop(result_file, status_log_file):
    """The main testing script"""
    datasets = ["ACIDS"]
    models = ["DeepSense", "TransformerV4"]
    learn_frameworks = ["SimCLR", "MoCo", "CMC", "Cosmo", "MTSS", "MoCoFusion"]
    tasks = ["vehicle_classification", "terrain_classification", "speed_classification"]
    label_ratios = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]

    # check status before testing
    for dataset in datasets:
        for model in models:
            for task in tasks:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios:
                        finetuned_flag = check_execution_flags(
                            status_log_file, dataset, model, task, learn_framework, label_ratio
                        )

                        if not finetuned_flag:
                            print(
                                f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio} has not been finetuned yet."
                            )
                        else:
                            print(f"Testing {dataset}-{model}-{learn_framework}-{task}-{label_ratio}.")
                            # set args
                            args = parse_base_args("test")
                            args.dataset = dataset
                            args.model = model
                            args.learn_framework = learn_framework
                            args.task = task
                            args.label_ratio = label_ratio
                            args.stage = "finetune"
                            args = set_auto_params(args)

                            # get result
                            classifier_loss, acc, f1 = test(args)
                            result = {
                                f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}": {
                                    "loss": classifier_loss,
                                    "acc": acc,
                                    "f1": f1,
                                },
                            }
                            update_finetune_result(result, result_file)


if __name__ == "__main__":
    status_log_file = "/home/sl29/FoundationSense/result/finetune_status.json"
    result_file = "/home/sl29/FoundationSense/result/finetune_result.json"

    start = time.time()
    test_loop(result_file, status_log_file)
    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .4f} seconds")
