import os
import json

from collections import OrderedDict


def init_execution_flags(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios):
    """Init the log of finetuning status."""
    if os.path.exists(status_log_file):
        status = json.load(open(status_log_file))
    else:
        status = {}

    for dataset in datasets:
        for model in models:
            for task in tasks[dataset]:
                for learn_framework in learn_frameworks:
                    for label_ratio in label_ratios:
                        if f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}" not in status:
                            status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"] = False

    with open(status_log_file, "w") as f:
        f.write(json.dumps(status, indent=4))


def update_execution_flags(status_log_file, dataset, model, task, learn_framework, label_ratio):
    """
    Update the status of finetuning status.
    """
    status = json.load(open(status_log_file))
    status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"] = True

    # sort the result
    status = dict(sorted(status.items()))

    with open(status_log_file, "w") as f:
        f.write(json.dumps(status, indent=4))


def reset_execution_flags(status_log_file, dataset, model, task, learn_framework, label_ratio):
    """
    Update the status of finetuning status.
    """
    status = json.load(open(status_log_file))
    status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"] = False

    with open(status_log_file, "w") as f:
        f.write(json.dumps(status, indent=4))


def check_execution_flags(status_log_file, dataset, model, task, learn_framework, label_ratio):
    """
    Check the status of finetuning status.
    """
    status = json.load(open(status_log_file))
    flag = (
        status[f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}"]
        if f"{dataset}-{model}-{learn_framework}-{task}-{label_ratio}" in status
        else False
    )

    return flag


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

    # sort the result
    complete_result = dict(sorted(complete_result.items()))

    with open(result_file, "w") as f:
        f.write(json.dumps(complete_result, indent=4))
