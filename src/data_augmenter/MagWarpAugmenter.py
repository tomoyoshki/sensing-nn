import torch
import torch.nn as nn

from random import random

from tsai.data.transforms import TSMagWarp
from tsai.data.core import TSTensor


class MagWarpAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["mag_warp"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.warp_func = TSMagWarp(magnitude=self.config["magnitude"], order=self.config["order"])

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the scaling augmenter. Operate in the time domain.
        split_idx: 0 for training set and 1 for validation set.
        magnitude: the strength of the warping function, relative ratio for multiplication.
        order: the number of knots in the warping.
        x: [b, c, i, s]
        Return: Same shape as x.
        """
        aug_loc_inputs = {}
        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                if random() < self.config["prob"]:
                    mod_input = org_loc_inputs[loc][mod].clone()
                    b, c, i, s = mod_input.shape
                    mod_input = torch.reshape(mod_input, (b, c, i * s))
                    aug_loc_inputs[loc][mod] = self.warp_func(TSTensor(mod_input), split_idx=0).reshape(b, c, i, s).data
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]

                # print((aug_loc_inputs[loc][mod] == org_loc_inputs[loc][mod]).sum().item())

        return aug_loc_inputs, labels
