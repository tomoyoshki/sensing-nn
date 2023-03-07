import os
import json
import time
import subprocess

from collections import OrderedDict
from output_utils.schedule_log_utils import init_execution_flags, update_execution_flags, check_execution_flags


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
    Schedule the finetuning jobs.
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
                for task in tasks:
                    for learn_framework in learn_frameworks:
                        for label_ratio in label_ratios:
                            # check if the job is done
                            if check_execution_flags(
                                status_log_file, dataset, model, task, learn_framework, label_ratio
                            ):
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
                                "-batch_size=128",
                                f"-label_ratio={label_ratio}",
                                f"-gpu={cuda_device}",
                            ]
                            # cmd = ["python", "random_process.py"]
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
    ]
    tasks = ["vehicle_classification", "terrain_classification", "speed_classification"]
    label_ratios = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]

    # hardware
    cuda_device_slots = {0: 2, 1: 2, 2: 2, 3: 2}

    # for logging
    status_log_file = "/home/sl29/FoundationSense/result/finetune_status.json"
    schedule_loop(status_log_file)
