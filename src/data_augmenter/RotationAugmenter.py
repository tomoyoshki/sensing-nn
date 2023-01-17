import torch
import torch.nn as nn

from random import random


class RotationAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """mixup and cutmix augmentation, does nothing if both has alpha 0"""
        super().__init__()
        self.args = args
        self.config = args.dataset_config["jitter"]
        self.noise_position = "time"
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.init_value_range()

    def forward(self, org_loc_inputs, labels):
        """
        TODO: Implement the rotation augmenter.
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

    def rotation_2d(x, ang=90):
        x_aug = np.empty(x.shape)
        if ang == 0:
            x_aug = x
        elif ang == 90:
            x_aug[:, 0] = -x[:, 1]
            x_aug[:, 1] = x[:, 0]
        elif ang == 180:
            x_aug = -x
        elif ang == 270:
            x_aug[:, 0] = x[:, 1]
            x_aug[:, 1] = -x[:, 0]
        else:
            print("Wrong input for rotation!")
        return x_aug
