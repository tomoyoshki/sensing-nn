import os
import os
import json
import time

import numpy as np

from test import test
from params.base_params import parse_base_args
from params.params_util import set_auto_params
from output_utils.schedule_log_utils import check_execution_flags, reset_execution_flags, update_finetune_result


def test_loop(result_file, status_log_file):
    """The main testing script"""
    datasets = ["ACIDS", "Parkland"]
    models = ["TransformerV4", "DeepSense"]
    learn_frameworks = [
        "SimCLR",
        "MoCo",
        "CMC",
        "Cosmo",
        "MTSS",
        "ModPred",
        "MoCoFusion",
        "SimCLRFusion",
        "ModPredFusion",
        "Cocoa",
    ]
    tasks = ["vehicle_classification", "terrain_classification", "speed_classification", "distance_classification"]
    label_ratios = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]

    # check status before testing
    for dataset in datasets:
        for model in models:
            for task in tasks:
                # skip useless task
                if (dataset == "ACIDS" and task == "distance_classification") or (
                    dataset == "Parkland" and task == "terrain_classification"
                ):
                    continue

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
                            # get result
                            try:
                                try:
                                    # set args
                                    args = parse_base_args("test")
                                    args.dataset = dataset
                                    args.model = model
                                    args.learn_framework = learn_framework
                                    args.task = task
                                    args.label_ratio = label_ratio
                                    args.stage = "finetune"
                                    args = set_auto_params(args)

                                    # eval the model
                                    classifier_loss, acc, f1 = test(args)
                                    result = {
                                        f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}": {
                                            "loss": classifier_loss,
                                            "acc": acc,
                                            "f1": f1,
                                        },
                                    }
                                except KeyboardInterrupt:
                                    print("Excution interrupted by user, terminating ...")
                                    return
                            except:
                                """No model available, reset the execution flag"""
                                print(f"Resetting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}.")
                                reset_execution_flags(
                                    status_log_file, dataset, model, task, learn_framework, label_ratio
                                )
                                continue

                            # update result
                            update_finetune_result(result, result_file)


if __name__ == "__main__":
    status_log_file = "/home/sl29/FoundationSense/result/finetune_status.json"
    result_file = "/home/sl29/FoundationSense/result/finetune_result.json"

    start = time.time()
    test_loop(result_file, status_log_file)
    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .4f} seconds")
