import torch.nn as nn

from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.TransformerV4 import TransformerV4
from models.NEWMODEL import NEWMODEL


def init_backbone_model(args):
    """Automatically select the model according to args."""
    if args.model == "DeepSense":
        classifier = DeepSense(args, self_attention=False)
    elif args.model == "TransformerV4":
        classifier = TransformerV4(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    elif args.model == "NEWMODEL":
        classifier = NEWMODEL(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")

    # move the model to the device
    classifier = classifier.to(args.device)

    return classifier

def init_loss_func(args):
    """Initialize the loss function according to the config."""
    if args.train_mode == "supervised" or args.stage == "finetune":
        if "regression" in args.task:
            loss_func = nn.MSELoss()
        else:
            loss_func = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Invalid train mode provided: {args.train_mode}")

    return loss_func
