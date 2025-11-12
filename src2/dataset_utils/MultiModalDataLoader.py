import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal sensing data (vehicle classification).
    
    Loads individual samples from .pt files as needed (lazy loading) to avoid memory overflow.
    Each .pt file contains a dictionary with 'data' and 'label' keys.
    
    Data structure:
        - data: dict[location][modality] = Tensor
        - label: dict with 'vehicle_type' key
    """
    
    def __init__(self, index_file, num_classes=None):
        """
        Initialize the dataset.
        
        Args:
            index_file (str): Path to file containing list of sample file paths
            num_classes (int, optional): Number of classes (required for balanced sampling)
        """
        self.num_classes = num_classes
        
        # Load sample file paths from index file
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        self.sample_files = list(np.loadtxt(index_file, dtype=str))
        logging.info(f"Initialized dataset with {len(self.sample_files)} samples")
    
    def compute_sample_weights_for_balanced_sampling(self):
        """
        Compute sample weights for balanced sampling based on class distribution.
        
        This is useful when you have imbalanced classes (e.g., 1000 samples of class A, 
        but only 100 samples of class B). Balanced sampling ensures each class is 
        equally represented during training by giving higher weight to minority classes.
        
        Call this method before creating the dataloader if you want balanced sampling.
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be provided for balanced sampling")
        
        sample_labels = []
        label_count = [0 for _ in range(self.num_classes)]
        
        logging.info("Computing sample weights for balanced sampling...")
        for idx in range(len(self.sample_files)):
            _, label, _ = self.__getitem__(idx)
            label_idx = label.item()
            sample_labels.append(label_idx)
            label_count[label_idx] += 1
        
        # Calculate weights: inverse of class frequency
        # More rare classes get higher weights
        self.sample_weights = []
        for sample_label in sample_labels:
            self.sample_weights.append(1.0 / label_count[sample_label])
        
        logging.info(f"Label distribution: {label_count}")
        logging.info(f"Sample weights computed for balanced sampling")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        """
        Load a single sample.
        
        Returns:
            data (dict): Multi-modal data dict[location][modality] = Tensor
            label (Tensor): Vehicle type label
            idx (int): Sample index
        """
        # Load sample from disk (lazy loading)
        sample_path = self.sample_files[idx]
        
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        
        try:
            sample = torch.load(sample_path, weights_only=True)
        except Exception as e:
            logging.error(f"Error loading {sample_path}: {e}")
            raise
        
        data = sample["data"]
        
        # Extract vehicle_type label from the label dictionary
        if isinstance(sample["label"], dict):
            if "vehicle_type" not in sample["label"]:
                raise KeyError(f"'vehicle_type' not found in sample labels. Available keys: {list(sample['label'].keys())}")
            label = sample["label"]["vehicle_type"]
        else:
            # If label is not a dict, assume it's directly the vehicle type
            label = sample["label"]
        
        return data, label, idx


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders from configuration.
    
    Args:
        config (dict): Configuration dictionary loaded from YAML
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Extract vehicle classification configuration
    batch_size = config.get("batch_size")
    num_workers = config.get("num_workers")
    use_balanced_sampling = config.get("use_balanced_sampling")
    if "vehicle_classification" not in config:
        raise ValueError("'vehicle_classification' not found in config")
    
    task_config = config["vehicle_classification"]
    num_classes = task_config.get("num_classes")
    
    # Get index file paths
    train_index_file = task_config.get("train_index_file")
    val_index_file = task_config.get("val_index_file")
    test_index_file = task_config.get("test_index_file")
    
    # Validate index files exist
    if not train_index_file or not os.path.exists(train_index_file):
        raise FileNotFoundError(f"Train index file not found: {train_index_file}")
    if not val_index_file or not os.path.exists(val_index_file):
        raise FileNotFoundError(f"Val index file not found: {val_index_file}")
    if not test_index_file or not os.path.exists(test_index_file):
        raise FileNotFoundError(f"Test index file not found: {test_index_file}")
    
    # Create datasets
    logging.info("Creating datasets...")
    
    train_dataset = MultiModalDataset(
        index_file=train_index_file,
        num_classes=num_classes
    )
    
    val_dataset = MultiModalDataset(
        index_file=val_index_file,
        num_classes=num_classes
    )
    
    test_dataset = MultiModalDataset(
        index_file=test_index_file,
        num_classes=num_classes
    )
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    
    # Training dataloader with optional balanced sampling
    if use_balanced_sampling:
        train_dataset.compute_sample_weights_for_balanced_sampling()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        logging.info("Using balanced sampling for training")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    # Val and test dataloaders (no shuffling or balancing)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logging.info(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    logging.info(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    logging.info(f"Test samples: {len(test_dataset)}, batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def create_single_dataloader(config, split="train", batch_size=32, num_workers=4, 
                             use_balanced_sampling=False):
    """
    Create a single dataloader for a specific split (train, val, or test).
    
    Args:
        config (dict): Configuration dictionary loaded from YAML
        split (str): Data split - 'train', 'val', or 'test'
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        use_balanced_sampling (bool): Whether to use balanced sampling (only for training)
    
    Returns:
        DataLoader: PyTorch dataloader
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    # Extract vehicle classification configuration
    if "vehicle_classification" not in config:
        raise ValueError("'vehicle_classification' not found in config")
    
    task_config = config["vehicle_classification"]
    num_classes = task_config.get("num_classes")
    
    # Get index file path
    index_file = task_config.get(f"{split}_index_file")
    
    if not index_file or not os.path.exists(index_file):
        raise FileNotFoundError(f"{split} index file not found: {index_file}")
    
    # Create dataset
    dataset = MultiModalDataset(
        index_file=index_file,
        num_classes=num_classes
    )
    
    # Create dataloader with optional balanced sampling (only for training)
    if use_balanced_sampling and split == "train":
        dataset.compute_sample_weights_for_balanced_sampling()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        logging.info(f"Using balanced sampling for {split}")
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    logging.info(f"{split.capitalize()} samples: {len(dataset)}, batches: {len(dataloader)}")
    
    return dataloader

