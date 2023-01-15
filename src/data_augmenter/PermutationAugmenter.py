import torch
import torch.nn as nn

from random import random


class PermutationAugmenter(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["permutation"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

    def forward(self, org_loc_inputs, labels):
        """
        Fake forward function of the permutation augmenter.
        x: [b, c, i, s]
        Return: Same shape as x. All samples are permuted in the same order.
        """
        aug_loc_inputs = {}
        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                if random() < self.config["prob"]:
                    mod_input = org_loc_inputs[loc][mod]
                    rand_time_order = torch.randperm(mod_input.shape[2])
                    aug_loc_inputs[loc][mod] = mod_input[:, :, rand_time_order, :]
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]

        return aug_loc_inputs, labels
