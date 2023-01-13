import torch
import torch.nn as nn

from input_utils.mixup_utils import Mixup


class MixupAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.mixup_func = Mixup(**args.dataset_config["mixup"])

    def forward(self, org_loc_inputs, labels):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        aug_loc_inputs, aug_labels = self.mixup_func(org_loc_inputs, labels, self.args.dataset_config)

        return aug_loc_inputs, aug_labels
