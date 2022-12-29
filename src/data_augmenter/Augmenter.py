import numpy as np
from data_augmenter.NoAugmenter import NoAugmenter
from data_augmenter.MissAugmenter import MissAugmenter
from data_augmenter.NoiseAugmenter import NoiseAugmenter
from data_augmenter.SeparateAugmenter import SeparateAugmenter


class Augmenter:
    def __init__(self, args) -> None:
        """This function is used to setup the data augmenter.

        Args:
            model (_type_): _description_
        """
        self.args = args
        self.mode = self.args.train_mode if self.args.option == "train" else self.args.inference_mode
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        print(f"=\t[Option]: {args.option}, mode: {self.mode}, stage: {args.stage}")

        if args.augmenter == "NoiseAugmenter":
            print(f"[Data augmenter config]: {args.miss_generator}-{args.noise_mode}")
        elif args.augmenter == "NoAugmenter":
            print(f"=\t{args.augmenter}")
        else:
            print(f"[Miss augmenter config]: {args.miss_generator}")

        # Step 1: Init the missing modality generator
        if args.augmenter == "SeparateAugmenter":
            self.augmenter = SeparateAugmenter(args)
        elif args.augmenter == "MissAugmenter":
            self.augmenter = MissAugmenter(args)
        elif args.augmenter == "NoiseAugmenter":
            self.augmenter = NoiseAugmenter(args)
        elif args.augmenter == "NoAugmenter":
            self.augmenter = NoAugmenter(args)
        else:
            raise Exception(f"Invalid **augmenter** provided: {args.augmenter}")

    def augment_forward(self, org_loc_inputs):
        """
        Add noise to the input_dict depending on the noise position.
        We only add noise to the time domeain, but not the feature level.
        NOTE: The noise is always generated at the time input level.
        """
        augmented_loc_inputs = dict()
        gt_loc_augmented_ids = dict()
        for loc in self.locations:
            loc_augmented_input, gt_miss_ids = self.augmenter(loc, org_loc_inputs[loc])
            augmented_loc_inputs[loc] = loc_augmented_input
            gt_loc_augmented_ids[loc] = gt_miss_ids

        return augmented_loc_inputs, gt_loc_augmented_ids

    def train(self):
        """Set all components to train mode"""
        self.augmenter.train()

    def eval(self):
        """Set all components to eval mode"""
        self.augmenter.eval()

    def to(self, device):
        """Move all components to the target device"""
        self.augmenter.to(device)
