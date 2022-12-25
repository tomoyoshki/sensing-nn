import torch
import torch.nn as nn
import numpy as np

from general_utils.tensor_utils import miss_ids_to_masks_feature, miss_ids_to_masks_input
from input_utils.normalize import all_value_ranges


class NoiseAugmenter(nn.Module):
    def __init__(self, args) -> None:
        """Separate missing modality generator"""
        super().__init__()
        self.args = args
        self.sensors = args.dataset_config["num_sensors"]
        self.candidate_ids = list(range(self.sensors))
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]

        # For inference: get the ids of missing modalities.
        self.loc_miss_ids = dict()
        self.loc_avl_ids = dict()
        for loc in self.locations:
            self.loc_miss_ids[loc] = []
            self.loc_avl_ids[loc] = []
            for i, mod in enumerate(self.modalities):
                if mod in args.miss_modalities:
                    self.loc_miss_ids[loc].append(i)
                else:
                    self.loc_avl_ids[loc].append(i)

        # noise mode, we always add noise to the time domain.
        self.noise_mode = "no"
        self.noise_position = "time"

        # load the value range

        """Save the base noise std for training the noise detector"""
        self.init_value_range()

        # random noise generator
        self.candidate_counts = list(range(1, self.sensors))
        num_miss_cases = self.sensors - 1
        self.candidate_count_probs = np.ones(num_miss_cases) / num_miss_cases

    def set_noise_mode(self, mode):
        """Set the noise mode."""
        assert mode in {
            "no",
            "fixed_gaussian",
            "random_gaussian",
            "laplacian",
            "exponential",
            "permute",
            "zero",
            "mosaic",
        }
        self.noise_mode = mode

    def forward(self, loc, loc_input):
        """
        Fake forward function of the no miss modality generator.
        NOTE: We always add the noise to the time domain.
        x: [b, c, i, s]
        Return: x_out and gt_miss_masks; for mask, same shape as x, 1 means available, 0 means missing.
        """
        return self.forward_input(loc, loc_input)

    def forward_input(self, loc, loc_input):
        """Forward function of the miss modality generator at the input (time or freq) level.
        Args:
            x (_type_): dict of modality input with shape [b, c, i, spectrum]
        Return:
            x_out: the same dict from mod to input, but some missing modalities.
            gt_miss_ids: the ground truth missing modality ids.
        """
        noisy_loc_input = dict()
        gt_miss_masks = dict()
        b = list(loc_input.values())[0].shape[0]

        if self.noise_mode == "no":
            noisy_loc_input = {mod: loc_input[mod] for mod in self.modalities}
            gt_miss_ids = [[] for _ in range(b)]
            gt_miss_masks = {mod_id: torch.ones_like(loc_input[mod]) for mod_id, mod in enumerate(self.modalities)}
        else:
            target_shapes = [loc_input[mod].shape for mod in self.modalities]
            gt_miss_ids = self.generate_random_miss_ids(b)
            gt_miss_masks = miss_ids_to_masks_input(gt_miss_ids, self.sensors, target_shapes, self.args.device)
            for mod_id, mod in enumerate(self.modalities):
                noisy_loc_input[mod] = self.add_noise_by_mask(loc_input[mod], gt_miss_masks[mod_id], mod=mod)

        return noisy_loc_input, gt_miss_ids

    def forward_feature(self, loc, loc_input):
        """General forward function at the feature level"""
        b = loc_input.shape[0]

        if self.noise_mode == "no":
            gt_miss_ids = [[] for _ in range(b)]
            return loc_input, gt_miss_ids
        else:
            gt_miss_ids = self.generate_random_miss_ids(loc_input.shape[0])
            gt_miss_masks = miss_ids_to_masks_feature(gt_miss_ids, loc_input.shape, self.args.device)
            noisy_loc_input = self.add_noise_by_mask(loc_input, gt_miss_masks, loc=loc)

            return noisy_loc_input, gt_miss_ids

    def add_noise_by_mask(self, input, gt_miss_masks, loc=None, mod=None):
        """ Automatically add noise according to the given gt_miss_masks.
           All positions marked by **0** will be added noise.
        Args:
            x (_type_): [b, c, i, s]
            gt_miss_masks (_type_): _description_
        """
        if self.noise_mode == "fixed_gaussian":
            return self.forward_fixed_gaussian(input, gt_miss_masks, loc, mod)
        elif self.noise_mode == "random_gaussian":
            return self.forward_random_gaussian(input, gt_miss_masks, loc, mod)
        elif self.noise_mode == "mosaic":
            return self.forward_mosaic(input, gt_miss_masks, loc, mod)
        elif self.noise_mode == "laplacian":
            return self.forward_laplacian(input, gt_miss_masks, loc, mod)
        elif self.noise_mode == "exponential":
            return self.forward_exponenetial(input, gt_miss_masks, loc, mod)
        elif self.noise_mode == "permute":
            return self.forward_time_permute(input, gt_miss_masks)
        elif self.noise_mode == "zero":
            return self.forward_zero(input, gt_miss_masks)
        else:
            raise Exception(f"Invalid noise mode: {self.noise_mode}")

    def forward_fixed_gaussian(self, input, gt_miss_masks, loc=None, mod=None):
        """Generate the noise on fixed modalities with given std."""
        args = self.args

        # set the noise std multipler
        if args.noise_std_multipler >= 0:
            std_multipler = args.noise_std_multipler
        else:
            std_multipler = 4

        # derive the noise std
        if self.noise_position == "feature":
            noise_std = self.base_noise_stds[loc] * std_multipler
        else:
            noise_std = self.base_noise_stds[mod] * std_multipler

        # generate the noise
        noise = torch.randn(input.shape).to(self.args.device) * noise_std
        noisy_loc_input = input.detach() + noise * (torch.ones_like(gt_miss_masks) - gt_miss_masks)

        return noisy_loc_input

    def forward_random_gaussian(self, input, gt_miss_masks, loc=None, mod=None):
        """Generate the random noise on random range of std for a location/modality."""
        args = self.args

        # random noise std
        if self.noise_position == "feature":
            base_noise_std = self.base_noise_stds[loc]
        else:
            base_noise_std = self.base_noise_stds[mod]

        if args.dataset in ["RealWorld_HAR"]:
            noise_std = np.random.uniform(low=0.5 * base_noise_std, high=base_noise_std * 20, size=1).item()
        elif args.dataset in ["Parkland"]:
            """Parkland dataset is very sensitive so we use a smaller range."""
            noise_std = np.random.uniform(low=0.1 * base_noise_std, high=base_noise_std * 6, size=1).item()
        elif args.dataset in ["Parkland1107"]:
            if args.model == "DeenSense":
                noise_std = np.random.uniform(low=0.1 * base_noise_std, high=base_noise_std * 3, size=1).item()
            elif args.model == "ResNet":
                noise_std = np.random.uniform(low=0.1 * base_noise_std, high=base_noise_std * 15, size=1).item()
            else:
                noise_std = np.random.uniform(low=0.1 * base_noise_std, high=base_noise_std * 20, size=1).item()
        else:
            """Because WESAD is generally resilient to noise, we use a larger range of std."""
            noise_std = np.random.choice(np.array([1, 2, 3, 5, 10, 15, 20, 50, 100]) * base_noise_std, size=1).item()

        noise = torch.randn(input.shape).to(self.args.device) * noise_std
        noisy_loc_input = input.detach() + noise * (torch.ones_like(gt_miss_masks) - gt_miss_masks)

        return noisy_loc_input

    def forward_mosaic(self, input, gt_miss_masks, loc=None, mod=None):
        """Randomly generate noise from a uniform distribution."""
        args = self.args

        # set the noise std multipler
        if args.noise_std_multipler >= 0:
            std_multipler = args.noise_std_multipler
        else:
            std_multipler = 4

        # derive the noise std
        if self.noise_position == "feature":
            noise_std = self.base_noise_stds[loc] * std_multipler
        else:
            noise_std = self.base_noise_stds[mod] * std_multipler

        noise = (torch.rand_like(input) - 0.5) * noise_std
        noisy_loc_input = input.detach() + noise.detach() * (torch.ones_like(gt_miss_masks) - gt_miss_masks)
        if self.args.model == "Transformer":
            noisy_loc_input = nn.functional.layer_norm(noisy_loc_input, input.shape)

        return noisy_loc_input

    def forward_laplacian(self, input, gt_miss_masks, loc=None, mod=None):
        """Randomly generate noise from a laplacian distribution.
        Example: dist = torch.distributions.laplace.Laplace(mean, scale)
        """
        args = self.args

        # set the noise std multipler
        if args.noise_std_multipler >= 0:
            std_multipler = args.noise_std_multipler
        else:
            std_multipler = 4

        # derive the noise std
        if self.noise_position == "feature":
            noise_std = self.base_noise_stds[loc] * std_multipler
        else:
            noise_std = self.base_noise_stds[mod] * std_multipler

        # generate the noise
        dist = torch.distributions.laplace.Laplace(0, noise_std)
        noise = dist.sample(input.shape).to(self.args.device)

        # add noise to the input
        noisy_loc_input = input.detach() + noise * (torch.ones_like(gt_miss_masks) - gt_miss_masks)
        return noisy_loc_input

    def forward_exponenetial(self, input, gt_miss_masks, loc=None, mod=None):
        """
        Randomly generate noise from a laplacian distribution.
        Example: rate = 1 / scale
        dist = torch.distributions.exponential.Exponential(rate)
        noise = dist.sample()
        """
        args = self.args

        # set the noise std multipler
        if args.noise_std_multipler >= 0:
            std_multipler = args.noise_std_multipler
        else:
            std_multipler = 4

        # derive the noise std
        if self.noise_position == "feature":
            noise_std = self.base_noise_stds[loc] * std_multipler
        else:
            noise_std = self.base_noise_stds[mod] * std_multipler

        # generate the noise
        dist = torch.distributions.exponential.Exponential(1 / noise_std)
        noise = dist.sample(input.shape).to(self.args.device)

        # add noise to the input
        noisy_loc_input = input.detach() + noise * (torch.ones_like(gt_miss_masks) - gt_miss_masks)
        return noisy_loc_input

    # New noise functions added by Ruijie:
    def forward_scaling(self, input, gt_miss_masks, loc=None, mod=None):
        """
        Randomly generate scaling factor from a normal distribution.
        https://arxiv.org/pdf/1706.00527.pdf
        """
        args = self.args

        # random noise std
        if self.noise_position == "feature":
            base_noise_std = self.base_noise_stds[loc]
        else:
            base_noise_std = self.base_noise_stds[mod]

        if args.dataset in ["RealWorld_HAR"]:
            noise_std = np.random.uniform(low=0.5 * base_noise_std, high=base_noise_std * 20, size=1).item()
        elif args.dataset in ["Parkland"]:
            """Parkland dataset is very sensitive so we use a smaller range."""
            noise_std = np.random.uniform(low=0.1 * base_noise_std, high=base_noise_std * 6, size=1).item()
        else:
            """Because WESAD is generally resilient to noise, we use a larger range of std."""
            noise_std = np.random.choice(np.array([1, 5, 20, 50, 100]) * base_noise_std, size=1).item()

        # generate the noise
        factor = torch.normal(mean=1.0, std=noise_std, size=input.shape).to(self.args.device)
        noisy_loc_input = input.detach() * gt_miss_masks + input.detach() * factor * (
            torch.ones_like(gt_miss_masks) - gt_miss_masks
        )

        return noisy_loc_input

    def forward_permute_jitter(self, input, gt_miss_masks, loc=None, mod=None):
        """
        Step1: Randomly permute time intervals;
        Step2: Randomly jitter time signals.
        https://github.com/emadeldeen24/TS-TCC/blob/e57050c1457aa279a5113b52ce8b923278c7a0ba/dataloader/augmentations.py
        """
        args = self.args

        permuted_loc_input = self.forward_time_permute(input, gt_miss_masks)
        noisy_loc_input = self.forward_random_gaussian(permuted_loc_input, gt_miss_masks, loc, mod)
        return noisy_loc_input

    def forward_zero(self, input, gt_miss_masks):
        """Randomly mask some element of the modality as zero."""
        args = self.args

        # Randomly set some elements with ratio noise_std_multipler to zero
        noise = (torch.rand_like(input) > args.noise_std_multipler / 100).float() * input

        # generate the output
        noisy_loc_input = input.detach() * gt_miss_masks + noise.detach() * (
            torch.ones_like(gt_miss_masks) - gt_miss_masks
        )

        return noisy_loc_input

    def forward_time_permute(self, loc_input, gt_miss_masks):
        """Randomly permute the time intervals, only applicable to DeepSense and ."""
        permuted_x = torch.permute(loc_input, [3, 0, 1, 2])
        noise = []
        for mod_feature in permuted_x:
            rand_time_order = torch.randperm(loc_input.shape[2])
            noise.append(mod_feature[:, :, rand_time_order])
        noise = torch.stack(noise).permute([1, 2, 3, 0])

        # generate the output
        noisy_loc_input = loc_input.detach() * gt_miss_masks + noise.detach() * (
            torch.ones_like(gt_miss_masks) - gt_miss_masks
        )

        return noisy_loc_input

    def generate_random_miss_ids(self, batch_size):
        """Generate the random missing sensor IDs"""
        miss_ids = []
        sample_miss_count = np.random.choice(self.candidate_counts, size=1, p=self.candidate_count_probs)
        for _ in range(batch_size):
            sample_miss_ids = np.random.choice(self.candidate_ids, sample_miss_count, replace=False)
            sample_miss_ids.sort()
            miss_ids.append(sample_miss_ids)

        return miss_ids

    def init_value_range(self):
        """Initialize the value range for each sensor."""
        self.value_ranges = all_value_ranges
        args = self.args
        self.base_noise_stds = {}
        if self.noise_position == "feature":
            value_range = self.value_ranges[args.dataset][self.noise_position][args.model]
            for loc in value_range:
                self.base_noise_stds[loc] = value_range[loc] / 100
        else:
            value_range = self.value_ranges[args.dataset][self.noise_position]
            for mod in value_range:
                self.base_noise_stds[mod] = value_range[mod] / 100
