import os
import json
import time
import subprocess

from collections import OrderedDict
from output_utils.schedule_log_utils import init_execution_flags, update_execution_flags, check_execution_flags
from params.output_paths import find_most_recent_weight
from params.params_util import get_train_mode


def check_cuda_slot(status_log_file, subprocess_pool, cuda_device_utils):
    """
    Return a cuda slot.
    """
    new_pid_pool = {}
    for p in subprocess_pool:
        if p.poll() is not None:
            """Subprocess has finished, release the cuda slot."""
            cuda_device = subprocess_pool[p]["cuda_device"]
            cuda_device_utils[cuda_device] = max(0, cuda_device_utils[cuda_device] - 1)
            update_execution_flags(status_log_file, *subprocess_pool[p]["info"])
        else:
            new_pid_pool[p] = subprocess_pool[p]

    subprocess_pool = new_pid_pool

    return subprocess_pool, cuda_device_utils


def claim_cuda_slot(cuda_device_utils):
    """
    Claim a cuda slot.
    """
    assigned_cuda_device = -1

    for cuda_device in cuda_device_utils:
        if cuda_device_utils[cuda_device] < cuda_device_slots[cuda_device]:
            cuda_device_utils[cuda_device] += 1
            assigned_cuda_device = cuda_device
            break

    return assigned_cuda_device, cuda_device_utils


def schedule_loop(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios):
    """
    Schedule the finetune jobs.
    """
    start = time.time()

    # init the execution flags
    init_execution_flags(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios)
    subprocess_pool = {}
    cuda_device_utils = {device: 0 for device in cuda_device_slots}

    # schedule the jobs
    try:
        for dataset in datasets:
            for model in models:
                for task in tasks[dataset]:
                    for learn_framework in learn_frameworks:
                        for label_ratio in label_ratios:
                            # check if the job is done
                            if check_execution_flags(
                                status_log_file, dataset, model, task, learn_framework, label_ratio
                            ):
                                continue

                            # check if we have pretrained weight
                            newest_id, _ = find_most_recent_weight(
                                False, dataset, model, get_train_mode(learn_framework), learn_framework
                            )
                            if newest_id < 0:
                                print(f"Skip {dataset}-{model}-{learn_framework}-{task}-{label_ratio}")
                                continue

                            # wait until a valid cuda device is available
                            cuda_device = -1
                            while cuda_device == -1:
                                subprocess_pool, cuda_device_utils = check_cuda_slot(
                                    status_log_file,
                                    subprocess_pool,
                                    cuda_device_utils,
                                )
                                cuda_device, cuda_device_utils = claim_cuda_slot(cuda_device_utils)

                            # run the command
                            cmd = [
                                "python3",
                                "train.py",
                                f"-dataset={dataset}",
                                f"-learn_framework={learn_framework}",
                                "-stage=finetune",
                                f"-task={task}",
                                f"-model={model}",
                                f"-label_ratio={label_ratio}",
                                f"-gpu={cuda_device}",
                                f"-debug=False",
                            ]
                            print(cmd)
                            p = subprocess.Popen(
                                cmd,
                                shell=False,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT,
                            )
                            subprocess_pool[p] = {
                                "cuda_device": cuda_device,
                                "info": (dataset, model, task, learn_framework, label_ratio),
                            }
                            print(f"Assigned cuda device: {cuda_device} for PID: {p.pid}")
                            print(f"CUDA device util status: {cuda_device_utils} \n")

        # wait until all subprocesses are finished
        while True:
            finished = True
            for p in subprocess_pool:
                if p.poll() is None:
                    finished = False
                    break
                else:
                    update_execution_flags(status_log_file, *subprocess_pool[p]["info"])

            if finished:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt, killing active subprocesses...")
        for p in subprocess_pool:
            p.kill()
        os.system(f"pkill -f stage=finetune")

    end = time.time()
    print("-" * 80)
    print(f"Total time: {end - start: .3f} seconds.")


if __name__ == "__main__":
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
    label_ratios = [1.0, 0.1, 0.01]

    # hardware
    cuda_device_slots = {0: 2, 1: 2, 2: 2, 3: 2}

    # for logging
    status_log_file = "/home/sl29/FoundationSense/result/finetune_status.json"

    # scheduling loop
    schedule_loop(status_log_file, datasets, models, tasks, learn_frameworks, label_ratios)
