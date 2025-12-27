# Quick Start: Experiment Tracking

## TL;DR

After training, view all your experiments with:
```bash
python src2/experiment_manager.py
```

## What's New?

Your training runs now automatically log:
- ✅ Quantization method and bitwidth options
- ✅ All hyperparameters (optimizer, scheduler, learning rate, etc.)
- ✅ Final training and validation accuracies
- ✅ Everything to TensorBoard AND a summary YAML file

## Quick Commands

### View all experiments
```bash
python src2/experiment_manager.py
```

### Find your best model
```bash
python src2/experiment_manager.py --sort_by best_val_accuracy
```

### View only quantization experiments
```bash
python src2/experiment_manager.py --filter_quantization
```

### See detailed info for each experiment
```bash
python src2/experiment_manager.py --detailed
```

### Export to CSV for analysis
```bash
python src2/experiment_manager.py --export my_experiments.csv
```

## What Gets Logged?

**Basic Info:**
- Model name and variant
- Dataset
- Batch size, epochs
- Optimizer, scheduler, loss function

**Quantization (if enabled):**
- Quantization method (dorefa, any_precision, etc.)
- Bitwidth options (e.g., [2], [8,16,32])
- Training method (joint_quantization, vanilla_single_precision_training)
- Validation function (simple_validation, random_bitwidths)

**Results:**
- Final training accuracy
- Final validation accuracy
- Best validation accuracy

## Where to Find Things?

**Experiment summary:**
```
src2/experiments/<experiment_id>/experiment_summary.yaml
```

**TensorBoard logs:**
```bash
tensorboard --logdir=src2/experiments/<experiment_id>/tensorboard
```

**Compare all experiments in TensorBoard:**
```bash
tensorboard --logdir=src2/experiments
```

## Example Output

```
#    Experiment ID                  Model           Dataset      Quant  Method     Bitwidths       Epochs  Best Val Acc  Status      
------------------------------------------------------------------------------------------------------------------------------------------------------
1    20251223_015151_ResNet_resne   ResNet/resnet   Parkland     Yes    dorefa     [2]             50      0.8912        completed   
2    20251223_011806_ResNet_resne   ResNet/resnet   Parkland     Yes    dorefa     [8,16,32]       50      0.8654        completed   

Best experiment: 20251223_015151_ResNet_resnet18
  Best Val Accuracy: 0.8912
  Model: ResNet (resnet18)
  Quantization: dorefa with bitwidths [2]
```

## No Changes to Your Workflow!

Just run training as usual:
```bash
python src2/train_test/train.py \
    --model ResNet \
    --model_variant resnet18 \
    --yaml_path src2/data/Parkland.yaml \
    --quantization_method dorefa \
    --gpu 0
```

Everything is logged automatically! 🎉

## More Info

See `EXPERIMENT_TRACKING_README.md` for full documentation.

