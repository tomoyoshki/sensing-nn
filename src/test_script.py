import os
import json
import time
import sys
import getpass
import numpy as np

from test import test
from eval_knn import eval_knn
from params.base_params import parse_base_args
from params.params_util import set_auto_params
from output_utils.schedule_log_utils import check_execution_flags, reset_execution_flags, update_finetune_result


def test_loop(result_file, status_log_file, test_mode):
    """The main testing script"""
    datasets = ["ACIDS", "Parkland", "RealWorld_HAR", "PAMAP2"]
    models = ["TransformerV4", "DeepSense"]
    learn_frameworks = [
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
    tasks = {
        "ACIDS": ["vehicle_classification"],
        "Parkland": ["vehicle_classification"],
        "RealWorld_HAR": ["activity_classification"],
        "PAMAP2": ["activity_classification"],
    }
    label_ratios = [1, 0.1, 0.01]

    # check status before testing
    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios:
                        # check if the model has been finetuned
                        if test_mode == "finetune":
                            finetuned_flag = check_execution_flags(
                                status_log_file, dataset, model, task, learn_framework, label_ratio
                            )
                        else:
                            finetuned_flag = True

                        # evaluate the model
                        if not finetuned_flag:
                            print(
                                f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio} has not been finetuned yet."
                            )
                        else:
                            print(f"\nTesting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}.")
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
                                    args.debug = "false"
                                    args = set_auto_params(args)

                                    # eval the model
                                    classifier_loss, acc, f1 = test(args) if test_mode == "finetune" else eval_knn(args)

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
                            except Exception as e:
                                print("Error: ", e)
                                if test_mode == "finetune":
                                    print(f"Resetting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}.\n")
                                    reset_execution_flags(
                                        status_log_file, dataset, model, task, learn_framework, label_ratio
                                    )
                                continue

                            # update result
                            update_finetune_result(result, result_file)


if __name__ == "__main__":
    args = parse_base_args("test")
    if args.test_mode == "finetune":
        test_mode = "finetune"
    elif args.test_mode == "knn":
        test_mode = "knn"
    else:
        raise Exception(f"Invalid evaluation mode {args.eval_mode}")

    username = getpass.getuser()
    status_log_file = f"/home/{username}/FoundationSense/result/finetune_status.json"
    result_file = f"/home/{username}/FoundationSense/result/{test_mode}_result.json"

    start = time.time()
    test_loop(result_file, status_log_file, test_mode)
    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .4f} seconds")
