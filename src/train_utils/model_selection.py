import torch.nn as nn

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

# Predictive Learning utils
from models.MTSSModules import MTSS
from models.ModPredModules import ModPred

# loss functions
from models.loss import DINOLoss, SimCLRLoss, MoCoLoss, CMCLoss


def init_model(args):
    """Automatically select the model according to args."""
    if args.model == "DeepSense":
        if args.train_mode == "contrastive" and args.stage == "pretrain" and args.contrastive_framework in {"MoCo"}:
            return DeepSense
        elif args.train_mode == "contrastive" and args.contrastive_framework in {"CMC", "Cosmo"}:
            classifier = DeepSense_CMC(args)
        else:
            classifier = DeepSense(args, self_attention=False)
    elif args.model == "TransformerV4":
        if args.train_mode == "contrastive" and args.stage == "pretrain" and args.contrastive_framework in {"MoCo"}:
            return TransformerV4
        elif args.train_mode == "contrastive" and args.contrastive_framework in {"CMC", "Cosmo"}:
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


def init_predictive_framework(args, backbone_model):
    """
    Initialize the predictive framework according to args.
    """
    if args.predictive_framework == "MTSS":
        default_model = MTSS(args, backbone_model)
    elif args.predictive_framework == "ModPred":
        default_model = ModPred(args, backbone_model)
    else:
        raise NotImplementedError

    default_model = default_model.to(args.device)
    return default_model


def init_loss_func(args, train_dataloader):
    """Initialize the loss function according to the config."""
    if args.multi_class:
        loss_func = nn.BCELoss()
    else:
        if args.train_mode == "supervised" or args.stage == "finetune":
            loss_func = nn.CrossEntropyLoss()
        elif args.train_mode == "predictive":
            """Predictive pretraining only."""
            if args.predictive_framework == "MTSS":
                loss_func = nn.CrossEntropyLoss()
            elif args.predictive_framework == "ModPred":
                loss_func = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError(f"Loss function for {args.predictive_framework} yet implemented")
        elif args.train_mode == "contrastive":
            """Contrastive pretraining only."""
            if args.contrastive_framework == "DINO":
                loss_func = DINOLoss(args).to(args.device)
            elif args.contrastive_framework == "MoCo":
                loss_func = MoCoLoss(args).to(args.device)
            elif args.contrastive_framework in {"CMC", "Cosmo"}:
                loss_func = CMCLoss(args, len(train_dataloader.dataset)).to(args.device)
            elif args.contrastive_framework in {"SimCLR"}:
                loss_func = SimCLRLoss(args).to(args.device)
            else:
                raise NotImplementedError(f"Loss function for {args.contrastive_framework} yet implemented")
        else:
            raise Exception(f"Invalid train mode provided: {args.train_mode}")

    return loss_func
