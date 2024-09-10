import os
import json
import logging


def check_paths(path_list):
    """
    Check the given path list.
    :param path_list:
    :return:
    """
    for p in path_list:
        if not os.path.exists(p):
            os.mkdir(p)


def remove_files(path_list):
    """
    Remove the given path list.
    :param path_list:
    :return:
    """
    for p in path_list:
        if os.path.exists(p):
            os.remove(p)


def set_model_weight_suffix(
    train_mode,
    learn_framework=None,
    task=None,
    label_ratio=None,
    miss_modalities=None,
    tag=None,
):
    """Automatically get the model path.

    Args:
        args (_type_): _description_
    """
    if train_mode == "supervised" and miss_modalities is not None and len(miss_modalities) > 0:
        suffix = f"{train_mode}-miss-"
        ordered_miss_modalities = list(miss_modalities)
        ordered_miss_modalities.sort()
        for mod in ordered_miss_modalities:
            suffix += f"-{mod}"
    else:
        """Other modes include: supervised, contrastive, predictive, and more..."""
        if train_mode == "supervised":
            suffix = f"supervised_{task}_{label_ratio}"
        elif train_mode in {"contrastive", "predictive", "generative"}:
            suffix = f"{train_mode}_{learn_framework}"
        else:
            raise Exception(f"Unknown train mode: {train_mode}")

    if tag is not None:
        suffix += f"-{tag}"

    return suffix

def find_most_recent_weight(
    debug,
    dataset,
    model,
    train_mode,
    learn_framework,
    task=None,
    label_ratio=None,
    return_suffix=False,
    tag=None,
):
    """Find the most recent weight path for the given (model, train_mode, framework).)"""
    # base model path
    base_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/weights"
    dataset_model_path = os.path.join(base_path, f"{dataset}_{model}")
    dataset_model_path += "_debug" if debug else ""

    # identify the proper suffix
    suffix = set_model_weight_suffix(train_mode, learn_framework, task, label_ratio, tag=tag)

    # find the most recent weight (training, finetuning, testing)
    newest_id = -1
    newest_weight = None
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
                newest_weight = os.path.join(dataset_model_path, weight)

    if return_suffix:
        return newest_id, newest_weight, suffix
    else:
        return newest_id, newest_weight


def set_model_weight_folder(args):
    """Automatically get the model path.

    Args:
        args (_type_): _description_
    """
    base_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/weights"
    dataset_model_path = os.path.join(base_path, f"{args.dataset}_{args.model}")
    dataset_model_path += "_debug" if args.debug else ""
    check_paths([base_path, dataset_model_path])

    # get the newest id matching the current config
    newest_id, newest_weight, suffix = find_most_recent_weight(
        args.debug,
        args.dataset,
        args.model,
        args.train_mode,
        args.learn_framework,
        args.task,
        args.label_ratio,
        return_suffix=True,
        tag=args.tag,
    )

    # set the weight path to avoid redundancy
    if args.option == "test" or args.stage in {"finetune", "alignment"}:
        if args.model_weight is not None:
            weight_folder = args.model_weight
        else:
            """Select the newest experiment in the given (dataset, model) config."""
            if newest_id == -1:
                raise Exception(f"No existing model weights for {suffix}")
            else:
                weight_folder = newest_weight
    else:
        "Supervised training or self-supervised pretraining"
        weight_folder = os.path.join(dataset_model_path, f"exp{newest_id + 1}") + f"_{suffix}"
        check_paths([weight_folder])
        model_config = args.dataset_config[args.model]
        
        
        with open(os.path.join(weight_folder, "experiment_args.json"), "w") as f:
            args_var = vars(args)
            args_record = {}
            for k in args_var:
                if isinstance(args_var[k], str):
                    args_record[k] = args_var[k]

            f.write(json.dumps(args_record, indent=4))

        with open(os.path.join(weight_folder, "model_config.json"), "w") as f:
            f.write(json.dumps(model_config, indent=4))

        if args.train_mode != "supervised" and args.stage == "pretrain":
            framework_config_log = os.path.join(weight_folder, "learn_framework_config.json")
            with open(framework_config_log, "w") as f:
                f.write(json.dumps(args.dataset_config[args.learn_framework], indent=4))

    tag_suffix = ""
    tag_suffix = tag_suffix if args.tag is None else f"{args.tag}"

    
    # set log files
    if args.option == "train":
        if args.train_mode == "supervised":
            args.train_log_file = os.path.join(weight_folder, f"{args.tag_suffix}train_log.txt")
            args.tensorboard_log = os.path.join(weight_folder, f"{args.tag_suffix}train_events")
        else:
            if args.stage == "pretrain":
                args.train_log_file = os.path.join(weight_folder, f"{tag_suffix}pretrain_log.txt")
                args.tensorboard_log = os.path.join(weight_folder, f"pretrain_events")
            elif args.stage == "finetune":
                args.train_log_file = os.path.join(weight_folder, f"{args.task}{args.finetune_tag_suffix}log.txt")
                args.tensorboard_log = os.path.join(weight_folder, f"{args.task}{args.finetune_tag_suffix}events")
            elif args.stage == "alignment":
                args.train_log_file = os.path.join(weight_folder, f"{args.tag_suffix}_log.txt")
                args.tensorboard_log = os.path.join(weight_folder, f"{args.tag_suffix}_events")
            else:
                raise Exception(f"Invalid stage provided: {args.stage}")

        # delete old log file
        remove_files([args.train_log_file])

        # set logging config
        logging.basicConfig(
            level=logging.INFO, handlers=[logging.FileHandler(args.train_log_file), logging.StreamHandler()], force=True
        )

        logging.info(f"=\t[Model weights path]: {weight_folder}")

    if args.comments is not None:
        tag = "" if args.tag is None else f"_{args.tag}"
        tag = f"{tag}_{args.stage}"
        tag = f"{tag}_{args.finetune_tag}" if args.finetune_tag is not None else tag
        with open(os.path.join(weight_folder, f"{tag}_comments.txt"), "a") as f:
            f.write(args.comments)
        logging.info(f"=\t[Comments]: {args.comments}")

    args.weight_folder = weight_folder

    return args


def set_model_weight_file(args):
    """Automatically select the classifier weight during the testing"""

    if args.train_mode == "supervised":
        # finetune_suffix = f"_{args.label_ratio}_finetune" if args.stage == "finetune" else ""
        args.classifier_weight = os.path.join(
            args.weight_folder,
            f"{args.dataset}_{args.model}_{args.task}{args.tag_suffix}best.pt",
        )
    elif args.train_mode in {"contrastive", "predictive", "generative"}:
        if args.stage == "pretrain":
            args.classifier_weight = os.path.join(
                args.weight_folder,
                f"{args.dataset}_{args.model}_pretrain_best.pt",
            )
        else:
            args.classifier_weight = os.path.join(
                args.weight_folder,
                f"{args.dataset}_{args.model}_{args.task}{args.finetune_tag_suffix}best.pt",
            )
    else:
        raise Exception(f"Invalid training mode provided: {args.stage}")

    logging.info(f"=\t[Classifier weight file]: {os.path.basename(args.classifier_weight)}")

    return args

def set_finetune_weights(args):   
    """Automatically select the finetune weight during the testing"""
    best_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_{args.task}{args.finetune_tag_suffix}best.pt",
    )
    latest_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_{args.task}{args.finetune_tag_suffix}latest.pt",
    )

    return best_weight, latest_weight


def set_alignment_weights(args):    
    """Automatically select the alignment weight during the testing"""
    best_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_{args.learn_framework}{args.alignment_tag_suffix}_best.pt",
    )
    latest_weight = os.path.join(
        args.weight_folder,
        f"{args.dataset}_{args.model}_{args.learn_framework}{args.alignment_tag_suffix}_latest.pt",
    )

    return best_weight, latest_weight


def set_output_paths(args):
    """Set the output paths. Not the weight path, but intermediate results used for analysis.

    Args:
        args (_type_): _description_
    """
    result_root_path = f"{os.path.abspath(os.path.join(os.getcwd(), os.pardir))}/result"
    log_root_path = os.path.join(result_root_path, "log")
    args.log_path = os.path.join(log_root_path, f"{args.dataset}_{args.model}_{args.train_mode}")
    check_paths([result_root_path, log_root_path, args.log_path])

    return args
