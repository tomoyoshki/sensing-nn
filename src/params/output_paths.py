import os
from datetime import date
import json


def check_paths(path_list):
    """
    Check the given path list.
    :param path_list:
    :return:
    """
    for p in path_list:
        if not os.path.exists(p):
            os.mkdir(p)


def set_model_weight_folder(args):
    """Automatically get the model path.

    Args:
        args (_type_): _description_
    """
    base_path = f"/home/{args.username}/FoundationSense/weights"
    dataset_model_path = os.path.join(base_path, f"{args.dataset}_{args.model}")
    check_paths([dataset_model_path])

    # suffix for different modes, only related to the **train mode**
    if args.train_mode == "supervised" and len(args.miss_modalities) > 0:
        suffix = f"{args.train_mode}-miss-"
        ordered_miss_modalities = list(args.miss_modalities)
        ordered_miss_modalities.sort()
        for mod in ordered_miss_modalities:
            suffix += f"-{mod}"
    else:
        """Other modes include: supervised, contrastive, and more..."""
        if args.train_mode == "supervised":
            suffix = f"supervised_{args.task}_{args.label_ratio}"
        elif args.train_mode == "contrastive":
            suffix = f"contrastive_{args.contrastive_framework}"
        elif args.train_mode == "MAE":
            suffix = f"mae_{args.model}"
        else:
            raise Exception(f"Unknown train mode: {args.train_mode}")

    # get the newest id matching the current config
    newest_id = -1
    existing_weights = os.listdir(dataset_model_path)
    for weight in existing_weights:
        # only check qualified weight with the required suffix
        weight_suffix = weight.split("_", 1)[-1]
        if suffix != weight_suffix:
            continue
        else:
            weight_id = int(weight.split("_")[0][3:])
            if weight_id > newest_id:
                newest_id = weight_id
                newest_weight = weight

    # set the weight path to avoid redundancy
    if args.option == "test" or args.stage == "finetune":
        """Test or finetuning"""
        if args.model_weight is None:
            """Select the newest experiment in the given (dataset, model) config."""
            if newest_id == -1:
                raise Exception(f"No existing model weights for {suffix}")
            else:
                weight_folder = os.path.join(dataset_model_path, newest_weight)
        else:
            weight_folder = args.model_weight
    else:
        "Supervised training or self-supervised pretraining"
        weight_folder = os.path.join(dataset_model_path, f"exp{newest_id + 1}") + f"_{suffix}"
        check_paths([weight_folder])
        model_config = args.dataset_config[args.model]
        contrastive_framework_config = args.dataset_config[args.contrastive_framework]
        with open(os.path.join(weight_folder, "model_config.json"), "w") as f:
            f.write(json.dumps(model_config, indent=4))
        with open(os.path.join(weight_folder, "contrastive_framework_config.json"), "w") as f:
            f.write(json.dumps(contrastive_framework_config, indent=4))

    # set log files
    if args.option == "train":
        if args.train_mode == "supervised":
            args.train_log_file = os.path.join(weight_folder, f"train_log.txt")
            args.tensorboard_log = os.path.join(weight_folder, f"train_events")
        else:
            if args.stage == "pretrain":
                args.train_log_file = os.path.join(weight_folder, f"pretrain_log.txt")
                args.tensorboard_log = os.path.join(weight_folder, f"pretrain_events")
            else:
                args.train_log_file = os.path.join(
                    weight_folder,
                    f"{args.task}_{args.label_ratio}_{args.stage}_log.txt",
                )
                args.tensorboard_log = os.path.join(
                    weight_folder,
                    f"{args.task}_{args.label_ratio}_{args.stage}_events",
                )

    print(f"[Model weights path]: {weight_folder}")
    args.weight_folder = weight_folder

    return args


def set_model_weight_file(args):
    """Automatically select the classifier weight during the training/testing"""
    if args.train_mode == "supervised":
        args.classifier_weight = os.path.join(
            args.weight_folder,
            f"{args.dataset}_{args.model}_{args.task}_best.pt",
        )
    elif args.train_mode in {"contrastive"}:
        if args.stage == "pretrain":
            args.classifier_weight = os.path.join(
                args.weight_folder,
                f"{args.dataset}_{args.model}_pretrain_best.pt",
            )
        else:
            args.classifier_weight = os.path.join(
                args.weight_folder,
                f"{args.dataset}_{args.model}_{args.task}_{args.label_ratio}_finetune_best.pt",
            )
    elif args.train_mode == "MAE":
        args.classifier_weight = os.path.join(
            args.weight_folder,
            f"{args.dataset}_{args.model}_{args.task}_best.pt",
        )
    else:
        raise Exception(f"Invalid training mode provided: {args.stage}")

    print(f"[Classifier weight file]: {os.path.basename(args.classifier_weight)}")

    return args


def set_output_paths(args):
    """Set the output paths. Not the weight path, but intermediate results used for analysis.

    Args:
        args (_type_): _description_
    """
    log_root_path = f"/home/{args.username}/FoundationSense/result/log"
    args.log_path = os.path.join(log_root_path, f"{args.dataset}_{args.model}_{args.train_mode}")
    check_paths([args.log_path])

    return args
