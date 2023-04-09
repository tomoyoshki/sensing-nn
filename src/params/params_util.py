import os
import json
import torch
import getpass
import logging

from params.output_paths import set_model_weight_file, set_output_paths, set_model_weight_folder
from input_utils.yaml_utils import load_yaml


def get_username():
    """The function to automatically get the username."""
    username = getpass.getuser()

    return username


def str_to_bool(flag):
    """
    Convert the string flag to bool.
    """
    if flag.lower() == "true":
        return True
    else:
        return False


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f"Torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = f"cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    print(s)

    return torch.device(arg)


def get_train_mode(learn_framework):
    """
    Automatically set the train mode according to the learn_framework.
    NOTE: Add the learn framework to this register when adding a new learn framework.
    """
    learn_framework_register = {
        "SimCLR": "contrastive",
        "SimCLRFusion": "contrastive",
        "MoCo": "contrastive",
        "MoCoFusion": "contrastive",
        "Cosmo": "contrastive",
        "CMC": "contrastive",
        "CMCV2": "contrastive",
        "Cocoa": "contrastive",
        "TNC": "contrastive",
        "MTSS": "predictive",
        "ModPred": "predictive",
        "ModPredFusion": "predictive",
        "MAE": "generative",
        "no": "supervised",
    }

    if learn_framework in learn_framework_register:
        train_mode = learn_framework_register[learn_framework]
    else:
        raise ValueError(f"Invalid learn_framework provided: {learn_framework}")

    return train_mode


def set_auto_params(args):
    """Automatically set the parameters for the experiment."""
    # gpu configuration
    if args.gpu is None:
        args.gpu = 0
    args.device = select_device(str(args.gpu))
    args.half = False  # half precision only supported on CUDA

    # retrieve the user name
    args.username = get_username()

    # parse the model yaml file
    dataset_yaml = f"./data/{args.dataset}.yaml"
    args.dataset_config = load_yaml(dataset_yaml)

    # verbose
    args.verbose = str_to_bool(args.verbose)
    args.count_range = str_to_bool(args.count_range)
    args.balanced_sample = str_to_bool(args.balanced_sample) and args.dataset in {"ACIDS", "Parkland_Miata"}
    args.sequence_sampler = True if args.learn_framework in {"CMCV2", "TS2Vec", "TNC"} else False
    args.debug = str_to_bool(args.debug)

    # threshold
    args.threshold = 0.5

    # dataloader config
    args.workers = 10

    # Sing-class problem or multi-class problem
    if args.dataset in {}:
        args.multi_class = True
    else:
        args.multi_class = False

    # process the missing modalities,
    if args.miss_modalities is not None:
        args.miss_modalities = set(args.miss_modalities.split(","))
        print(f"Missing modalities: {args.miss_modalities}")
    else:
        args.miss_modalities = set()

    # set the train mode
    args.train_mode = get_train_mode(args.learn_framework)
    print(f"Set train mode: {args.train_mode}")

    # set output path
    args = set_model_weight_folder(args)
    args = set_model_weight_file(args)
    args = set_output_paths(args)

    return args
