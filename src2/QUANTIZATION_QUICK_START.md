# DoReFa Quantization Quick Start Guide

## Running Training with Quantization

### Basic Command

**Option 1: Run from repository root** (Recommended)
```bash
cd /home/misra8/sensing-nn

python src2/train_test/train.py \
  --model ResNet \
  --model_variant resnet18 \
  --yaml_path src2/data/Parkland.yaml \
  --quantization_method dorefa \
  --gpu 0
```

**Option 2: Run from train_test directory**
```bash
cd /home/misra8/sensing-nn/src2/train_test

python train.py \
  --model ResNet \
  --model_variant resnet18 \
  --yaml_path ../data/Parkland.yaml \
  --quantization_method dorefa \
  --gpu 0
```

Note: The YAML path is different depending on where you run from!

### What Happens During Training

1. **Model Creation**: ResNet is created with QuanConv layers instead of standard Conv2d
2. **Quantization Setup**: All QuanConv layers are configured with DoReFa quantizers
3. **Joint Quantization Training**: For each batch:
   - 2 forward passes with different random bitwidth configurations
   - Loss is averaged across these passes
   - Single backward pass with averaged gradient
4. **Random Bitwidths Validation**: At validation time:
   - Model is tested with 3 random bitwidth configurations
   - Statistics reported: mean, min, max, std accuracy/loss
   - Best model selected based on mean accuracy

### Expected Output

```
Epoch [1/50]
Training with joint quantization (batch_size=2)
  Train Loss: 1.8234, Train Acc: 0.3421
  
Validating with 3 random bitwidth configurations...
  Config 1/3: Acc=0.3512, Loss=1.7823
  Config 2/3: Acc=0.3487, Loss=1.7901
  Config 3/3: Acc=0.3534, Loss=1.7756
  
Random Bitwidths Validation Statistics:
  Accuracy - Mean: 0.3511, Min: 0.3487, Max: 0.3534, Std: 0.0019
  Loss - Mean: 1.7827, Min: 1.7756, Max: 1.7901, Std: 0.0060
  
Best model saved! (Val Acc: 0.3511)
```

## Configuration

### Current Settings (in Parkland.yaml)

```yaml
quantization:
  enable: True
  Conv: "QuanConv"
  
  dorefa:
    bitwidth_options: [8, 16, 32]        # Bitwidths to sample from
    weight_quantization: "dorefa"         # Weight quantization method
    activation_quantization: "dorefa"     # Activation quantization method
    training_method: "joint_quantization" # Training strategy
    validation_function: "random_bitwidths" # Validation strategy
    switchable_clipping: True             # For PACT (future)
    sat_weight_normalization: False       # Weight normalization
  
  joint_quantization:
    joint_quantization_batch_size: 2      # Forward passes per batch
  
  random_bitwidths:
    num_bitwidths: 3                      # Validation configurations
```

### Modifying Settings

To change bitwidth options:
```yaml
dorefa:
  bitwidth_options: [4, 6, 8]  # Use lower bitwidths
```

To increase joint quantization intensity:
```yaml
joint_quantization:
  joint_quantization_batch_size: 4  # More forward passes
```

To test more configurations during validation:
```yaml
random_bitwidths:
  num_bitwidths: 5  # More comprehensive validation
```

## Testing

Run the integration tests to verify everything works:

```bash
cd /home/misra8/sensing-nn
python src2/test_quantization_integration.py
```

Expected output:
```
============================================================
QUANTIZATION INTEGRATION TESTS
============================================================

✓ PASS: QuanConv Methods
✓ PASS: Configuration Loading
✓ PASS: Bitwidth Setting
✓ PASS: Quantization Setup
✓ PASS: Model Creation

Total: 5/5 tests passed
✓ All tests passed!
```

## Disabling Quantization

To run standard (non-quantized) training:

1. Set `enable: False` in config:
```yaml
quantization:
  enable: False
```

2. Or use a different config file without quantization

## Troubleshooting

### Issue: "YAML configuration file not found"
**Solution**: You're running from the wrong directory or using the wrong path.

- **If running from repository root** (`/home/misra8/sensing-nn`):
  ```bash
  python src2/train_test/train.py --yaml_path src2/data/Parkland.yaml ...
  ```

- **If running from train_test** (`/home/misra8/sensing-nn/src2/train_test`):
  ```bash
  python train.py --yaml_path ../data/Parkland.yaml ...
  ```

- **If unsure, use absolute path**:
  ```bash
  python train.py --yaml_path /home/misra8/sensing-nn/src2/data/Parkland.yaml ...
  ```

### Issue: "Quantization method not found"
**Solution**: Make sure `--quantization_method` matches a key in `config['quantization']`

### Issue: "Bitwidth not in options"
**Solution**: Ensure all bitwidth options in your config are consistent

### Issue: "QuanConv has no attribute setup_quantize_funcs"
**Solution**: Verify you're using the QuanConv from `src2/models/QuantModules.py`

## Monitoring Training

### TensorBoard

Training automatically logs to TensorBoard.

**From repository root:**
```bash
tensorboard --logdir=src2/experiments/
```

**From train_test directory:**
```bash
tensorboard --logdir=../experiments/
```

**Or use the exact path shown in the training output** (recommended - it's displayed at the start of training)

Look for these metrics:
- `Loss/train` and `Loss/val`
- `Accuracy/train` and `Accuracy/val`
- `Validation/mean_acc`, `Validation/std_acc`
- `Validation/min_acc`, `Validation/max_acc`

### Log Files

**From repository root:**
```bash
tail -f src2/experiments/<experiment_id>/logs/train_quantization.log
```

**From train_test directory:**
```bash
tail -f ../experiments/<experiment_id>/logs/train_quantization.log
```

**Or use wildcard to find the latest:**
```bash
# From repo root
tail -f src2/experiments/*/logs/train_quantization.log

# From train_test directory
tail -f ../experiments/*/logs/train_quantization.log
```

## Model Checkpoints

Models are saved to (paths relative to repository root):
- `src2/experiments/<experiment_id>/models/best_model.pth` - Best validation accuracy
- `src2/experiments/<experiment_id>/models/last_epoch.pth` - Latest epoch

The exact paths are displayed in the training output.

Checkpoint includes:
- Model state dict
- Optimizer state dict
- Validation accuracy and loss
- Full config
- Quantization method used

## Advanced Usage

### Using PACT Quantization

Modify config:
```yaml
dorefa:
  activation_quantization: "pact"  # Use PACT instead of DoReFa
  switchable_clipping: True
```

### Using LSQ/LSQPlus

```yaml
dorefa:
  weight_quantization: "lsq"
  activation_quantization: "lsq"
```

### Mixed Methods

```yaml
dorefa:
  weight_quantization: "dorefa"
  activation_quantization: "pact"  # Mix DoReFa weights with PACT activations
```

## Performance Tips

1. **GPU Memory**: Joint quantization increases memory usage. Reduce `joint_quantization_batch_size` if OOM
2. **Training Speed**: More joint passes = slower training. Start with 2, increase if convergence is poor
3. **Bitwidth Range**: Wider range [4, 6, 8, 16, 32] = better mixed-precision but slower training

## Example Full Training Command

**From repository root:**
```bash
cd /home/misra8/sensing-nn

# Full featured training with quantization
python src2/train_test/train.py \
  --model ResNet \
  --model_variant resnet18 \
  --yaml_path src2/data/Parkland.yaml \
  --quantization_method dorefa \
  --loss cross_entropy \
  --gpu 0

# View results in TensorBoard
tensorboard --logdir=src2/experiments/

# Check the logs
tail -f src2/experiments/*/logs/train_quantization.log
```

**From train_test directory:**
```bash
cd /home/misra8/sensing-nn/src2/train_test

# Full featured training with quantization
python train.py \
  --model ResNet \
  --model_variant resnet18 \
  --yaml_path ../data/Parkland.yaml \
  --quantization_method dorefa \
  --loss cross_entropy \
  --gpu 0

# View results in TensorBoard (from train_test dir)
tensorboard --logdir=../experiments/

# Check the logs (from train_test dir)
tail -f ../experiments/*/logs/train_quantization.log
```

