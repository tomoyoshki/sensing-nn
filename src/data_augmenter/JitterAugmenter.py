import torch
import torch.nn as nn

from random import random
from input_utils.normalize import all_value_ranges


class JitterAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["jitter"]
        self.noise_position = "time"
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.init_value_range()

    def forward(self, org_loc_inputs, labels=None):
        """
        Fake forward function of the no miss modality generator.
        x: [b, c, i, s]
        Return: Same shape as x, 1 means available, 0 means missing.
        """
        aug_loc_inputs = {}
        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                if random() < self.config["prob"]:
                    mod_input = org_loc_inputs[loc][mod]
                    noise = torch.randn(mod_input.shape).to(self.args.device) * self.base_noise_stds[mod]
                    aug_loc_inputs[loc][mod] = mod_input + noise
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]

        return aug_loc_inputs, labels

    def init_value_range(self):
        """Initialize the value range for each sensor."""
        self.value_ranges = all_value_ranges
        self.base_noise_stds = {}
        value_range = self.value_ranges[self.args.dataset][self.noise_position]
        for mod in value_range:
            self.base_noise_stds[mod] = value_range[mod] / 100 * self.config["std_in_percent"]
