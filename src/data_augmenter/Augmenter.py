import logging
import torch
import numpy as np
from data_augmenter.NoAugmenter import NoAugmenter
from data_augmenter.MissAugmenter import MissAugmenter
from data_augmenter.NoiseAugmenter import NoiseAugmenter
from data_augmenter.SeparateAugmenter import SeparateAugmenter
from data_augmenter.MixupAugmenter import MixupAugmenter


class Augmenter:
    def __init__(self, args) -> None:
        """This function is used to setup the data augmenter.
        We define a list of augmenters according to the config file, and run the augmentation sequentially.
        Args:
            model (_type_): _description_
        """
        self.args = args
        self.train_flag = True
        self.mode = self.args.train_mode if self.args.option == "train" else self.args.inference_mode
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        logging.info(f"=\t[Option]: {args.option}, mode: {self.mode}, stage: {args.stage}")

        # augmenter pool
        augmenter_pool = {
            "no": NoAugmenter,
            "miss": MissAugmenter,
            "noise": NoiseAugmenter,
            "mixup": MixupAugmenter,
        }

        # load the augmenters
        self.augmenter_names = args.dataset_config[args.model]["augmenters"]
        self.augmenters = []
        for augmenter_name in self.augmenter_names:
            if augmenter_name not in augmenter_pool:
                raise Exception(f"Invalid augmenter provided: {augmenter_name}")
            else:
                self.augmenters.append(augmenter_pool[augmenter_name](args))
                logging.info(f"=\t[Loaded augmenter]: {augmenter_name}")

    def forward(self, time_loc_inputs, labels):
        """
        Add noise to the input_dict depending on the noise position.
        We only add noise to the time domeain, but not the feature level.
        """
        # move to target device
        time_loc_inputs, labels = self.move_to_target_device(time_loc_inputs, labels)

        # time --> freq domain with FFT
        freq_loc_inputs = self.fft_preprocess(time_loc_inputs)

        # data augmentation
        aug_freq_loc_inputs, aug_labels = freq_loc_inputs, labels
        if self.train_flag:
            for augmenter in self.augmenters:
                aug_freq_loc_inputs, aug_labels = augmenter(aug_freq_loc_inputs, aug_labels)

        return aug_freq_loc_inputs, aug_labels

    def move_to_target_device(self, time_loc_inputs, labels):
        """Move both the data and labels to the target device"""
        target_device = self.args.device

        for loc in time_loc_inputs:
            for mod in time_loc_inputs[loc]:
                time_loc_inputs[loc][mod] = time_loc_inputs[loc][mod].to(target_device)

        labels = labels.to(target_device)

        return time_loc_inputs, labels

    def fft_preprocess(self, time_loc_inputs):
        """Run FFT on the time-domain input.
        time_loc_inputs: [b, c, i, s]
        freq_loc_inputs: [b, c, i, s]
        """
        freq_loc_inputs = dict()

        for loc in time_loc_inputs:
            freq_loc_inputs[loc] = dict()
            for mod in time_loc_inputs[loc]:
                loc_mod_freq_output = torch.fft.fft(time_loc_inputs[loc][mod], dim=-1)
                loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
                loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
                b, c1, c2, i, s = loc_mod_freq_output.shape
                loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
                freq_loc_inputs[loc][mod] = loc_mod_freq_output

        return freq_loc_inputs

    def train(self):
        """Set all components to train mode"""
        self.train_flag = True
        for augmenter in self.augmenters:
            augmenter.train()

    def eval(self):
        """Set all components to eval mode"""
        self.train_flag = False
        for augmenter in self.augmenters:
            augmenter.eval()

    def to(self, device):
        """Move all components to the target device"""
        for augmenter in self.augmenters:
            augmenter.to(device)
