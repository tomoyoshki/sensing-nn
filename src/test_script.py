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
from params.finetune_configs import *
from output_utils.schedule_log_utils import check_execution_flag, reset_execution_flag, update_finetune_result


def test_loop(result_file, status_log_file, test_mode):
    """The main testing script"""
    # check status before testing
    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios:
                        for run_id in range(runs):
                            # only once for label_ratio = 1.0
                            if label_ratio == 1.0 and run_id > 0:
                                continue

                            # check if the model has been finetuned
                            finetuned_flag = (
                                True
                                if test_mode == "knn"
                                else check_execution_flag(
                                    status_log_file, dataset, model, task, learn_framework, label_ratio, run_id
                                )
                            )
                            if not finetuned_flag:
                                print(
                                    f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id} not finetuned yet."
                                )
                                continue

                            # evaluate the model
                            print(f"\nTesting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}.")
                            try:
                                try:
                                    # set args
                                    args = parse_base_args("test")
                                    args.dataset = dataset
                                    args.model = model
                                    args.learn_framework = learn_framework
                                    args.task = task
                                    args.label_ratio = label_ratio
                                    args.finetune_run_id = run_id
                                    args.stage = "finetune"
                                    args.debug = "false"
                                    args = set_auto_params(args)

                                    # eval the model
                                    classifier_loss, acc, f1 = test(args) if test_mode == "finetune" else eval_knn(args)

                                    tmp_result = {
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
                                    print(
                                        f"Resetting {dataset}-{model}-{learn_framework}-{task}-{label_ratio}-exp{run_id}.\n"
                                    )
                                    reset_execution_flag(
                                        status_log_file, dataset, model, task, learn_framework, label_ratio, run_id
                                    )
                                continue

                            # update result
                            update_finetune_result(tmp_result, result_file)


def calc_mean_result(result_file):
    """Calculate the mean result"""
    out_result = {}
    out_file = result_file.replace(".json", "_mean.json")

    with open(result_file, "r") as f:
        org_result = json.load(f)

    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios:
                        # check result
                        if f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}" not in org_result:
                            print(f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio} not in result.")
                            continue

                        tmp_result = org_result[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"]
                        tmp_acc = np.array(tmp_result["acc"])
                        tmp_f1 = np.array(tmp_result["f1"])
                        tmp_loss = np.array(tmp_result["loss"])

                        out_result[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"] = {
                            "acc": {
                                "mean": tmp_acc.mean(),
                                "std": tmp_acc.std(),
                            },
                            "f1": {
                                "mean": tmp_f1.mean(),
                                "std": tmp_f1.std(),
                            },
                            "loss": {
                                "mean": tmp_loss.mean(),
                                "std": tmp_loss.std(),
                            },
                        }

    with open(out_file, "w") as f:
        json.dump(out_result, f, indent=4)


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
    # Step 1: test the finetuned models
    test_loop(result_file, status_log_file, test_mode)

    # Step 2: calculate the mean result
    calc_mean_result(result_file)
    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .4f} seconds")
