from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3
from models.TransformerV4 import TransformerV4
from models.TransformerV4_CMC import TransformerV4_CMC
from models.DeepSense_CMC import DeepSense_CMC


def init_model(args):
    """Automatically select the model according to args."""
    if args.model == "DeepSense":
        if args.train_mode == "contrastive" and args.stage == "pretrain" and args.contrastive_framework in {"MoCo"}:
            return DeepSense
        elif args.train_mode == "contrastive" and args.contrastive_framework == "CMC":
            classifier = DeepSense_CMC(args)
        else:
            classifier = DeepSense(args, self_attention=False)
    elif args.model == "TransformerV4":
        if args.train_mode == "contrastive" and args.stage == "pretrain" and args.contrastive_framework in {"MoCo"}:
            return TransformerV4
        elif args.train_mode == "contrastive" and args.contrastive_framework == "CMC":
            classifier = TransformerV4_CMC(args)
        else:
            classifier = TransformerV4(args)
    elif args.model == "ResNet":
        classifier = ResNet(args)
    else:
        raise Exception(f"Invalid model provided: {args.model}")

    # move the model to the device
    classifier = classifier.to(args.device)

    return classifier
