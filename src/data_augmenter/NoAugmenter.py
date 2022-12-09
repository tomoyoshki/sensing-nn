from tkinter.messagebox import NO
import torch
import torch.nn as nn
import numpy as np


class NoAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """None missing modality generator"""
        super().__init__()
        self.args = args

    def forward(self, loc, loc_input):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        return self.forward_input(loc_input)

    def forward_input(self, loc_input):
        """
        The fake forward function at the input level for a single location.
        Args:
            x: dict from mod to input data of shape [b, c, i, spectrum]
        Return:
            loc_x: dict from mod to input data of the same shape, but with missing modalities.
            loc_gt_miss_ids:[list of miss ids for each sample]
        """
        loc_gt_miss_masks = dict()
        b = None

        for mod_id, mod in enumerate(loc_input):
            loc_gt_miss_masks[mod_id] = torch.ones_like(loc_input[mod])
            if b is None:
                b = loc_input[mod].shape[0]

        # generate the miss ids
        loc_gt_miss_ids = [[] for _ in range(b)]

        return loc_input, loc_gt_miss_ids

    def forward_feature(self, loc_input):
        """
        The fake forward function at the feature level.
        Args:
            x: [b, c, i, sensors]
        Return:
            the same shape as x, and the corresponding miss masks
        """
        b = loc_input.shape[0]
        loc_gt_miss_ids = [[] for _ in range(b)]

        return loc_input, loc_gt_miss_ids
