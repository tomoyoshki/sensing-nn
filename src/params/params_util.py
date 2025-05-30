import os
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
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)"  # bytes to MB
        arg = f"cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    logging.info(s)

    return torch.device(arg)


def get_train_mode(learn_framework):
    """
    Automatically set the train mode according to the learn_framework.
    NOTE: Add the learn framework to this register when adding a new learn framework.
    """
    learn_framework_register = {
        "no": "supervised",
    }

    if learn_framework in learn_framework_register:
        train_mode = learn_framework_register[learn_framework]
    else:
        raise ValueError(f"Invalid learn_framework provided: {learn_framework}")

    return train_mode


def set_task(args):
    """
    Set the default task according to the dataset.
    """
    task_default_task = {
        "ACIDS": "vehicle_classification",
        "Parkland": "vehicle_classification",
        "RealWorld_HAR": "activity_classification",
        "PAMAP2": "activity_classification",
    }

    task = task_default_task[args.dataset] if args.task is None else args.task

    logging.info("Task: ", task)

    return task


def set_batch_size(args):
    """
    Automatically set the batch size for different (dataset, task, train_mode).
    """
    if args.batch_size is None:
        if args.dataset == "PPG":
            if args.stage == "pretrain":
                args.batch_size = 512
            else:
                args.batch_size = 256
        else:
            if args.stage == "pretrain":
                args.batch_size = 256
            else:
                args.batch_size = 128

    return args


def set_tag(args):
    """
    Automatically set the training configs according to the given tage. Mainly used in the ablation study.
    """
    if args.tag == "noTemp":
        args.dataset_config["CMCV2"]["rank_loss_weight"] = 0
    elif args.tag == "noOrth":
        args.dataset_config["CMCV2"]["orthogonal_loss_weight"] = 0
    elif args.tag == "noPrivate":
        args.dataset_config["CMCV2"]["private_contrastive_loss_weight"] = 0
        args.dataset_config["CMCV2"]["orthogonal_loss_weight"] = 0

    return args

def set_model_weight_file_suffix(args):
    
    finetune_tag_suffix = ""
    finetune_tag_suffix = finetune_tag_suffix if args.finetune_run_id is None else f"_exp{args.finetune_run_id}{finetune_tag_suffix}"
    finetune_tag_suffix = finetune_tag_suffix if args.finetune_tag is None else f"_{args.finetune_tag}{finetune_tag_suffix}"
    finetune_tag_suffix = f"_{args.label_ratio}{finetune_tag_suffix}"
    finetune_tag_suffix = finetune_tag_suffix if args.finetune_set is None else f"_{args.finetune_set}{finetune_tag_suffix}"
    finetune_tag_suffix = finetune_tag_suffix if args.modality is None else f"_{args.modality}{finetune_tag_suffix}"
    
    finetune_tag_suffix = f"_finetune{finetune_tag_suffix}"
    
    tag_suffix = f""
    tag_suffix = f"{tag_suffix}{finetune_tag_suffix}" if args.stage in {"finetune"} else tag_suffix
    
    args.tag_suffix = tag_suffix
    args.finetune_tag_suffix = finetune_tag_suffix
    return args


def set_data_config(args):
    args.num_class = args.dataset_config[args.task]["num_classes"]
    return args


def set_auto_params(args, lambda_type=None, lambda_weight=None, margin_value=None):
    """Automatically set the parameters for the experiment."""
    # gpu configuration
    if args.gpu is None:
        args.gpu = 0
    args.device = select_device(str(args.gpu))
    args.half = False  # half precision only supported on CUDA

    # retrieve the user name
    args.username = get_username()

    # set downstream task
    args.task = set_task(args)

    # parse the model yaml file
    dataset_yaml = f"./data/{args.dataset}.yaml"
    args.dataset_config = load_yaml(dataset_yaml)
    
    if lambda_type is not None and lambda_weight is not None:
        args.dataset_config[args.learn_framework][lambda_type] = lambda_weight
    
    if margin_value is not None:
        args.dataset_config[args.learn_framework]["inter_rank_margin"] = margin_value

    if args.modality is not None:
        args.dataset_config["modality_names"] = [args.modality]

    # verbose
    args.use_gcq_data = str_to_bool(args.use_gcq_data)
    args.use_gcq_align = str_to_bool(args.use_gcq_align)
    args.verbose = str_to_bool(args.verbose)
    args.count_range = str_to_bool(args.count_range)
    args.balanced_sample = str_to_bool(args.balanced_sample)
    args.sequence_sampler = True if args.learn_framework in {"CMCV2", "TS2Vec", "TS2Vec", "TNC", "TSTCC", "InfoMAE", "MultMod"} else False
    args.debug = str_to_bool(args.debug)
    args.output_conf = str_to_bool(args.output_conf)
    args.seq_dataset = str_to_bool(args.seq_dataset)

    # threshold
    args.threshold = 0.5

    # dataloader config
    args.workers = 10

    # Sing-class problem or multi-class problem
    if args.dataset in {} or "multiclass" in args.finetune_tag:
        args.multi_class = True
    else:
        args.multi_class = False

    # process the missing modalities,
    if args.miss_modalities is not None:
        args.miss_modalities = set(args.miss_modalities.split(","))
    else:
        args.miss_modalities = set()

    # set the train mode
    args.train_mode = get_train_mode(args.learn_framework)
    
    if args.train_mode == "supervised":
        args.stage = "train"

    # set batch size
    args = set_batch_size(args)

    # set tag
    args = set_tag(args)

    # set output path
    args = set_model_weight_file_suffix(args)
    args = set_model_weight_folder(args)
    args = set_model_weight_file(args)
    args = set_output_paths(args)


    # additional config
    args = set_data_config(args)
    
    if args.option in {"train"}:
        logging.info(f"=\t[Finetune Tag Suffix]: {args.finetune_tag_suffix}")
        logging.info(f"=\t[Tag suffix]: {args.tag_suffix}")
    else:
        print(f"=\t[Finetune Tag Suffix]: {args.finetune_tag_suffix}")
        print(f"=\t[Tag suffix]: {args.tag_suffix}")

    return args