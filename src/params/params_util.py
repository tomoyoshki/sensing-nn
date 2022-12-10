import os
import json
import torch
import getpass

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
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f"Torch {torch.__version__} "  # string
    device = str(device).strip().lower().replace("cuda:", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    # set GPU config
    cuda = (not cpu) and torch.cuda.is_available()
    if cuda:
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
    else:
        s += "CPU"

    if not newline:
        s = s.rstrip()

    print("-----------------------------GPU Setting--------------------------------")
    print(s)

    return torch.device("cuda:0" if cuda else "cpu")


def auto_select_augmenter(args):
    """Automatically select the data augmenter, mainly for separate training mode."""
    if args.option == "train":
        if args.inference_mode == "separate":
            args.augmenter = "SeparateAugmenter"
    else:
        if args.inference_mode == "separate":
            args.augmenter = "SeparateAugmenter"
        else:
            args.augmenter = "NoAugmenter"

    return args


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
    args.save_emb = str_to_bool(args.save_emb)
    args.elastic_mod = str_to_bool(args.elastic_mod)
    args.test_noisy_parkland = str_to_bool(args.test_noisy_parkland)
    args.test_wind_parkland = str_to_bool(args.test_wind_parkland)

    # threshold
    args.threshold = 0.5

    # dataloader config
    args.workers = 10

    # count values for noise generator
    args.count_range = False

    # triplet batch size
    # args.triplet_batch_size = int(args.batch_size / 3)

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

    # automatically set the miss generator and the detector
    args = auto_select_augmenter(args)

    # set output path
    args = set_model_weight_folder(args)
    args = set_model_weight_file(args)
    args = set_output_paths(args)

    return args
