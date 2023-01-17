from torch import optim as optim


def define_optimizer(args, parameters):
    """Define the optimizer."""
    if args.train_mode == "supervised":
        optimizer_config = args.dataset_config[args.model]["optimizer"]
    elif args.train_mode == "contrastive" and args.stage == "pretrain":
        optimizer_config = args.dataset_config[args.contrastive_framework]["pretrain_optimizer"]
    elif args.train_mode == "contrastive" and args.stage == "finetune":
        optimizer_config = args.dataset_config[args.contrastive_framework]["finetune_optimizer"]
    else:
        raise Exception("Optimizer not defined.")
    optimizer_name = optimizer_config["name"]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            parameters,
            lr=optimizer_config["start_lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented.")

    return optimizer
