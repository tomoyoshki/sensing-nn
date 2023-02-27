from models.ResNet import ResNet
from models.DeepSense import DeepSense
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2
from models.TransformerV3 import TransformerV3
from models.TransformerV4 import TransformerV4
from models.TransformerV4_CMC import TransformerV4_CMC
from models.DeepSense_CMC import DeepSense_CMC

# Contrastive Learning utils
from models.DINOModules import DINO
from models.SimCLRModules import SimCLR
from models.MoCoModule import MoCoWrapper
from models.CMCModules import CMC
from models.CosmoModules import Cosmo


def init_model(args):
    """Automatically select the model according to args."""
    if args.model == "DeepSense":
        if args.train_mode == "contrastive" and args.stage == "pretrain" and args.contrastive_framework in {"MoCo"}:
            return DeepSense
        elif args.train_mode == "contrastive" and args.contrastive_framework in {"CMC", "Cosmo"}:
            classifier = DeepSense_CMC(args)
        elif args.train_mode == "MAE":
            classifier = DeepSense_CMC(args)
        else:
            classifier = DeepSense(args, self_attention=False)
    elif args.model == "TransformerV4":
        if args.train_mode == "contrastive" and args.stage == "pretrain" and args.contrastive_framework in {"MoCo"}:
            return TransformerV4
        elif (
            args.train_mode == "contrastive" and args.contrastive_framework in {"CMC", "Cosmo"}
        ) or args.train_mode == "MAE":
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


def init_contrastive_framework(args, backbone_model):
    # model config
    if args.contrastive_framework == "SimCLR":
        default_model = SimCLR(args, backbone_model)
    elif args.contrastive_framework == "DINO":
        default_model = DINO(args, backbone_model)
    elif args.contrastive_framework == "MoCo":
        default_model = MoCoWrapper(args, backbone_model)
    elif args.contrastive_framework == "CMC":
        default_model = CMC(args, backbone_model)
    elif args.contrastive_framework == "Cosmo":
        default_model = Cosmo(args, backbone_model)
    else:
        raise NotImplementedError
    default_model = default_model.to(args.device)
    return default_model
