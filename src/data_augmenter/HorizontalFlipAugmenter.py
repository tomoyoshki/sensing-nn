import torch
import torch.nn as nn

from random import random


class HorizontalFlipAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["horizontal_flip"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the horizontal augmenter. Operate in the time domain.
        x: [b, c, i, s]
        Return: Same shape as x. The input is flipped horizontally with probability self.config["prob"].
        """
        aug_loc_inputs = {}
        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                if random() < self.config["prob"]:
                    aug_loc_inputs[loc][mod] = torch.flip(org_loc_inputs[loc][mod], dims=[2, 3])
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]

        return aug_loc_inputs, labels
