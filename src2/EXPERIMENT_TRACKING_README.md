# Experiment Tracking and Management

This document describes the experiment tracking features added to the training pipeline.

## Overview

The training pipeline now includes comprehensive experiment tracking with:
1. **TensorBoard Hyperparameter Logging** - All hyperparameters and final metrics logged to TensorBoard
2. **Experiment Summary Files** - YAML files with key information about each experiment
3. **Experiment Manager Script** - Command-line tool to view and compare all experiments

## Features Added

### 1. TensorBoard Hyperparameter Logging

When you run training, the following information is automatically logged to TensorBoard:

**Model Information:**
- Model name and variant
- Dataset name
- Batch size
- Number of epochs

**Optimizer & Training:**
- Learning rate
- Weight decay
- Optimizer type (AdamW, Adam, SGD)
- Scheduler type (cosine, step, multistep)
- Loss function

**Quantization (if enabled):**
- Quantization method (dorefa, any_precision, etc.)
- Bitwidth options (e.g., [2], [8,16,32])
- Training method (joint_quantization, vanilla_single_precision_training)
- Validation function (simple_validation, random_bitwidths)
- Weight quantization method
- Activation quantization method

**Final Metrics:**
- Final training accuracy
- Final validation accuracy
- Best validation accuracy

To view these in TensorBoard:
```bash
tensorboard --logdir=src2/experiments/<experiment_id>/tensorboard
```

Then navigate to the "HPARAMS" tab in TensorBoard to see all hyperparameters and metrics.

### 2. Experiment Summary Files

After training completes (or is interrupted), an `experiment_summary.yaml` file is automatically created in each experiment directory with the following structure:

```yaml
experiment_id: 20251223_015151_ResNet_resnet18
model: ResNet
model_variant: resnet18
dataset: Parkland
quantization_enabled: 'True'
quantization_method: dorefa
bitwidth_options: '[2]'
training_method: vanilla_single_precision_training
validation_function: simple_validation
num_epochs: 50
batch_size: 128
learning_rate: 0.0001
optimizer: AdamW
scheduler: cosine
loss_function: cross_entropy
final_train_accuracy: 0.9234
final_val_accuracy: 0.8876
best_val_accuracy: 0.8912
training_status: completed
```

The `training_status` field can be:
- `completed` - Training finished successfully
- `interrupted` - Training was interrupted by user (Ctrl+C)
- `failed` - Training failed with an error
- `no_summary` - Old experiment without summary file

### 3. Experiment Manager Script

The `experiment_manager.py` script provides a convenient way to view and compare all experiments.

#### Basic Usage

View all experiments:
```bash
python src2/experiment_manager.py
```

View all experiments from a specific directory:
```bash
python src2/experiment_manager.py --experiments_dir src2/experiments
```

#### Filtering

Show only quantization experiments:
```bash
python src2/experiment_manager.py --filter_quantization
```

Filter by model:
```bash
python src2/experiment_manager.py --filter_model ResNet
```

#### Sorting

Sort by best validation accuracy (highest first):
```bash
python src2/experiment_manager.py --sort_by best_val_accuracy
```

Sort by Average Validation Standard Deviation
```bash
python src2/experiment_manager.py --sort_by avg_val_std
```


Sort by experiment ID (chronological):
```bash
python src2/experiment_manager.py --sort_by experiment_id
```

Other sortable fields: `model`, `dataset`, `final_train_accuracy`, `final_val_accuracy`

#### Detailed View

Show detailed information for each experiment:
```bash
python src2/experiment_manager.py --detailed
```

This displays:
- Model and variant
- Dataset
- Training status
- All quantization settings (if enabled)
- Training hyperparameters
- Final metrics

#### Export to CSV

Export all experiments to a CSV file for analysis in Excel/Python:
```bash
python src2/experiment_manager.py --export experiments.csv
```

#### Combined Examples

Show only quantization experiments, sorted by accuracy, with detailed view:
```bash
python src2/experiment_manager.py --filter_quantization --sort_by best_val_accuracy --detailed
```

Export quantization experiments to CSV:
```bash
python src2/experiment_manager.py --filter_quantization --export quantization_experiments.csv
```

## Output Examples

### Compact Table View (Default)

```
======================================================================================================================================================
EXPERIMENT SUMMARY (14 experiments)
======================================================================================================================================================

#    Experiment ID                  Model           Dataset      Quant  Method     Bitwidths       Epochs  Best Val Acc  Status      
------------------------------------------------------------------------------------------------------------------------------------------------------
1    20251223_015151_ResNet_resne   ResNet/resnet   Parkland     Yes    dorefa     [2]             50      0.8912        completed   
2    20251205_012303_ResNet_resne   ResNet/resnet   Parkland     Yes    dorefa     N/A             50      N/A           no_summary  
...

======================================================================================================================================================

Total experiments: 14
Completed: 1
Quantization enabled: 14

Best experiment: 20251223_015151_ResNet_resnet18
  Best Val Accuracy: 0.8912
  Model: ResNet (resnet18)
  Quantization: dorefa with bitwidths [2]
======================================================================================================================================================
```

### Detailed View

```
1. 20251223_015151_ResNet_resnet18
----------------------------------------------------------------------------------------------------
  Model:                  ResNet (resnet18)
  Dataset:                Parkland
  Status:                 completed
  Quantization:           True
  Quantization Method:    dorefa
  Bitwidth Options:       [2]
  Training Method:        vanilla_single_precision_training
  Validation Function:    simple_validation
  Epochs:                 50
  Batch Size:             128
  Learning Rate:          0.0001
  Optimizer:              AdamW
  Scheduler:              cosine
  Loss Function:          cross_entropy
  Final Train Accuracy:   0.9234
  Final Val Accuracy:     0.8876
  Best Val Accuracy:      0.8912
```

## Integration with Training

The experiment tracking is fully integrated into the training pipeline. When you run:

```bash
python src2/train_test/train.py \
    --model ResNet \
    --model_variant resnet18 \
    --yaml_path src2/data/Parkland.yaml \
    --quantization_method dorefa \
    --gpu 0
```

The following happens automatically:
1. Hyperparameters are logged to TensorBoard at the start of training
2. Training proceeds normally with all metrics logged
3. At the end of training (or if interrupted), final metrics are logged to TensorBoard
4. An `experiment_summary.yaml` file is created with all key information
5. You can immediately use `experiment_manager.py` to view this experiment alongside others

## Tips

1. **Quick comparison**: Use the default table view to quickly scan all experiments
2. **Finding best model**: Use `--sort_by best_val_accuracy` to find your best performing model
3. **Analyzing quantization**: Use `--filter_quantization --detailed` to see all quantization experiments with full details
4. **Data analysis**: Export to CSV and use pandas/Excel for custom analysis
5. **TensorBoard comparison**: You can compare multiple experiments in TensorBoard by pointing it to the parent experiments directory:
   ```bash
   tensorboard --logdir=src2/experiments
   ```

## Backward Compatibility

The experiment manager works with both:
- **New experiments** (with `experiment_summary.yaml` files)
- **Old experiments** (without summary files - it reads from `config.yaml` instead)

Old experiments will show `training_status: no_summary` and will have `N/A` for final accuracies.

## File Locations

- **Training script**: `src2/train_test/train.py`
- **Experiment manager**: `src2/experiment_manager.py`
- **Experiments directory**: `src2/experiments/`
- **Summary files**: `src2/experiments/<experiment_id>/experiment_summary.yaml`
- **TensorBoard logs**: `src2/experiments/<experiment_id>/tensorboard/`

## Help

For full command-line options:
```bash
python src2/experiment_manager.py --help
```

