import sys
import logging
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    config = get_config()
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info("\nDataloaders created successfully!")
    
    augmenter = create_augmenter(config, augmentation_mode="no")
    logging.info("Augmenter created successfully!")
    
    for batch_idx, (data, labels, idx) in enumerate(train_loader):
        augmented_data, augmented_labels = apply_augmentation(augmenter, data, labels)
        logging.info(f"Batch {batch_idx}: Data augmented successfully")
        breakpoint()
        if batch_idx >= 2:
            break
    


if __name__ == "__main__":
    main()

