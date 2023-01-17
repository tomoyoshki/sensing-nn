import logging
import torch
import numpy as np
import random
from data_augmenter.NoAugmenter import NoAugmenter
from data_augmenter.MissAugmenter import MissAugmenter
from data_augmenter.MixupAugmenter import MixupAugmenter
from data_augmenter.JitterAugmenter import JitterAugmenter
from data_augmenter.PermutationAugmenter import PermutationAugmenter
from data_augmenter.ScalingAugmenter import ScalingAugmenter
from data_augmenter.NegationAugmenter import NegationAugmenter
from data_augmenter.HorizontalFlipAugmenter import HorizontalFlipAugmenter
from data_augmenter.ChannelShuffleAugmenter import ChannelShuffleAugmenter
from data_augmenter.TimeWarpAugmenter import TimeWarpAugmenter
from data_augmenter.MagWarpAugmenter import MagWarpAugmenter
from data_augmenter.TimeMaskAugmenter import TimeMaskAugmenter

from data_augmenter.FreqMaskAugmenter import FreqMaskAugmenter
from data_augmenter.PhaseShiftAugmenter import PhaseShiftAugmenter


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

        # load the time augmenters
        self.time_augmenter_pool = {
            "no": NoAugmenter,
            "miss": MissAugmenter,
            "mixup": MixupAugmenter,
            "jitter": JitterAugmenter,
            "permutation": PermutationAugmenter,
            "scaling": ScalingAugmenter,
            "negation": NegationAugmenter,
            "horizontal_flip": HorizontalFlipAugmenter,
            "channel_shuffle": ChannelShuffleAugmenter,
            "time_warp": TimeWarpAugmenter,
            "mag_warp": MagWarpAugmenter,
            "time_mask": TimeMaskAugmenter,
        }
        self.time_aug_names = args.dataset_config[args.model]["time_augmenters"]
        self.time_augmenters = []
        for aug_name in self.time_aug_names:
            if aug_name not in self.time_augmenter_pool:
                raise Exception(f"Invalid augmenter provided: {aug_name}")
            else:
                self.time_augmenters.append(self.time_augmenter_pool[aug_name](args))
                logging.info(f"=\t[Loaded time augmenter]: {aug_name}")

        # load the freq augmenters
        self.freq_augmenter_pool = {
            "no": NoAugmenter,
            "freq_mask": FreqMaskAugmenter,
            "phase_shift": PhaseShiftAugmenter,
        }
        self.freq_aug_names = args.dataset_config[args.model]["freq_augmenters"]
        self.freq_augmenters = []
        for aug_name in self.freq_aug_names:
            if aug_name not in self.freq_augmenter_pool:
                raise Exception(f"Invalid augmenter provided: {aug_name}")
            else:
                self.freq_augmenters.append(self.freq_augmenter_pool[aug_name](args))
                logging.info(f"=\t[Loaded frequency augmenter]: {aug_name}")

        # random augmenter pool
        self.aug_names = self.time_aug_names + self.freq_aug_names
        self.augmenters = self.time_augmenters + self.freq_augmenters

    def forward(self, time_loc_inputs, labels):
        """General interface for the forward function."""
        args = self.args
        if args.train_mode == "supervised":
            return self.forward_fixed(time_loc_inputs, labels)
        elif args.train_mode == "contrastive":
            if args.stage == "pretrain":
                return self.forward_random(time_loc_inputs, labels)
            else:
                return self.forward_noaug(time_loc_inputs, labels)
        else:
            raise Exception(f"Invalid train mode: {args.train_mode}")

    def forward_fixed(self, time_loc_inputs, labels):
        """
        Add noise to the input_dict depending on the noise position.
        We only add noise to the time domeain, but not the feature level.
        """
        # move to target device
        time_loc_inputs, labels = self.move_to_target_device(time_loc_inputs, labels)

        # time-domain augmentation
        aug_time_loc_inputs, aug_labels = time_loc_inputs, labels
        if self.train_flag:
            for augmenter in self.time_augmenters:
                aug_time_loc_inputs, aug_labels = augmenter(aug_time_loc_inputs, aug_labels)

        # time --> freq domain with FFT
        freq_loc_inputs = self.fft_preprocess(aug_time_loc_inputs)

        # freq-domain augmentation
        aug_freq_loc_inputs, aug_labels = freq_loc_inputs, labels
        if self.train_flag:
            for augmenter in self.freq_augmenters:
                aug_freq_loc_inputs, aug_labels = augmenter(aug_freq_loc_inputs, aug_labels)

        return aug_freq_loc_inputs, aug_labels

    def forward_random(self, time_loc_inputs, labels=None):
        """Randomly select one augmenter from both (time, freq) augmenter pool and apply it to the input."""
        # move to target device
        time_loc_inputs, labels = self.move_to_target_device(time_loc_inputs, labels)

        # select a random augmenter
        rand_aug_id = np.random.randint(len(self.aug_names))
        rand_aug_name = self.aug_names[rand_aug_id]
        rand_augmenter = self.augmenters[rand_aug_id]

        # time-domain augmentation
        aug_time_loc_inputs, aug_labels = time_loc_inputs, labels
        if self.train_flag and rand_aug_name in self.time_augmenter_pool:
            aug_time_loc_inputs, aug_labels = rand_augmenter(aug_time_loc_inputs, aug_labels)

        # time --> freq domain with FFT
        freq_loc_inputs = self.fft_preprocess(aug_time_loc_inputs)

        # freq-domain augmentation
        aug_freq_loc_inputs, aug_labels = freq_loc_inputs, labels
        if self.train_flag and rand_aug_name in self.freq_augmenter_pool:
            aug_freq_loc_inputs, aug_labels = rand_augmenter(aug_freq_loc_inputs, aug_labels)

        if labels is None:
            return aug_freq_loc_inputs
        else:
            return aug_freq_loc_inputs, aug_labels

    def forward_noaug(self, time_loc_inputs, labels):
        """
        Add noise to the input_dict depending on the noise position.
        We only add noise to the time domeain, but not the feature level.
        """
        # move to target device
        time_loc_inputs, labels = self.move_to_target_device(time_loc_inputs, labels)

        # time --> freq domain with FFT
        freq_loc_inputs = self.fft_preprocess(time_loc_inputs)

        return freq_loc_inputs, labels

    def move_to_target_device(self, time_loc_inputs, labels):
        """Move both the data and labels to the target device"""
        target_device = self.args.device

        for loc in time_loc_inputs:
            for mod in time_loc_inputs[loc]:
                time_loc_inputs[loc][mod] = time_loc_inputs[loc][mod].to(target_device)

        if not (labels is None):
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

    def eval(self):
        """Set all components to eval mode"""
        self.train_flag = False

    def to(self, device):
        """Move all components to the target device"""
        for augmenter in self.time_augmenters:
            augmenter.to(device)
