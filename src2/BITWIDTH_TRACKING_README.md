# Bitwidth Tracking and Visualization

## Overview

Added comprehensive average bitwidth tracking and visualization for quantization-aware training with random bitwidth validation.

## New Features

### 1. Average Bitwidth Calculation

A new function `get_average_bitwidth()` calculates the average bitwidth across all quantized Conv layers in the model:

```python
def get_average_bitwidth(model):
    """
    Calculate the average bitwidth across all quantized Conv layers.
    
    Returns:
        float: Average bitwidth across all layers
    """
```

This function:
- Finds all quantized Conv layers in the model
- Extracts the current bitwidth from each layer
- Returns the mean bitwidth across all layers

### 2. Enhanced `validate_random_bitwidths()` Function

The validation function now tracks and reports bitwidth statistics:

**What's Logged:**
- Average bitwidth for each random configuration
- Mean, min, max, and std deviation of bitwidths
- Correlation between bitwidth and accuracy

**Example Output:**
```
Config 1/4: Acc=0.8912, Loss=0.2156, Avg Bitwidth=12.45
Config 2/4: Acc=0.8654, Loss=0.2387, Avg Bitwidth=8.32
Config 3/4: Acc=0.9123, Loss=0.1987, Avg Bitwidth=18.76
Config 4/4: Acc=0.8876, Loss=0.2245, Avg Bitwidth=14.21

Random Bitwidths Validation Statistics:
  Accuracy - Mean: 0.8891, Min: 0.8654, Max: 0.9123, Std: 0.0189
  Loss - Mean: 0.2194, Min: 0.1987, Max: 0.2387, Std: 0.0162
  Bitwidth - Mean: 13.44, Min: 8.32, Max: 18.76, Std: 4.1234
```

### 3. TensorBoard Logging

All bitwidth metrics are automatically logged to TensorBoard:

**Scalar Metrics:**
- `Bitwidth/mean_bitwidth` - Average bitwidth across all configs
- `Bitwidth/std_bitwidth` - Standard deviation of bitwidths
- `Bitwidth/min_bitwidth` - Minimum bitwidth observed
- `Bitwidth/max_bitwidth` - Maximum bitwidth observed

These metrics are logged every epoch alongside accuracy and loss metrics.

### 4. Bitwidth vs Accuracy Visualization

A new plotting function creates comprehensive visualizations:

```python
def plot_bitwidth_vs_accuracy(bitwidths, accuracies, epoch, title):
    """
    Create scatter plot with error bars showing bitwidth vs accuracy.
    
    Features:
    - Individual configuration points (scatter)
    - Mean accuracy line across bitwidth bins
    - Error bars showing standard deviation
    - Min-max range indicators
    - Statistics text box
    """
```

**Plot Features:**
- **Scatter points**: Each random configuration plotted as a point
- **Mean line**: Red line connecting mean accuracies for each bitwidth bin
- **Error bars**: Standard deviation shown as error bars
- **Min-Max range**: Vertical lines showing full range
- **Statistics box**: Shows overall mean ± std for both accuracy and bitwidth
- **Legend**: Clear labeling of all elements

**When Logged:**
- Every 5 epochs
- Final epoch
- Available in TensorBoard under "Bitwidth_Analysis/bitwidth_vs_accuracy"

### 5. Visualization Examples

#### Single Bitwidth (Vanilla Training)
When using a fixed bitwidth, the plot shows:
- All configurations clustered at the same bitwidth
- Horizontal mean line with std deviation
- Focuses on accuracy variance

#### Multiple Bitwidths (Joint Quantization)
When using multiple bitwidths, the plot shows:
- Clear relationship between bitwidth and accuracy
- Bins for each bitwidth option
- Mean accuracy trend across different bitwidths
- Variance within each bitwidth configuration

## Usage

No code changes needed! The features activate automatically when:
1. Quantization is enabled
2. Validation function is set to `"random_bitwidths"`

### Example Training Command

```bash
python src2/train_test/train.py \
    --model ResNet \
    --model_variant resnet18 \
    --yaml_path src2/data/Parkland.yaml \
    --quantization_method any_precision \
    --gpu 0
```

### Viewing in TensorBoard

```bash
tensorboard --logdir=src2/experiments/<experiment_id>/tensorboard
```

Then navigate to:
- **SCALARS** tab → "Bitwidth" section for time-series plots
- **IMAGES** tab → "Bitwidth_Analysis" for the scatter plots

## Configuration

In your YAML config file, ensure you have:

```yaml
quantization:
  enable: True
  your_method:
    bitwidth_options: [8, 16, 32]  # Multiple bitwidths
    validation_function: "random_bitwidths"  # Enable this feature
    training_method: "joint_quantization"  # Or vanilla_single_precision_training

random_bitwidths:
  num_bitwidths: 4  # Number of random configs to test
```

## Interpreting Results

### What to Look For

1. **Bitwidth-Accuracy Correlation**
   - Positive correlation: Higher bitwidths → Higher accuracy (expected)
   - Flat correlation: Model robust across bitwidths
   - Non-monotonic: Some bitwidths work better than others

2. **Variance Analysis**
   - Low std deviation: Stable performance across configs
   - High std deviation: Configuration-sensitive model
   - Use error bars to judge stability

3. **Optimal Bitwidth**
   - Find the "knee point" where accuracy plateaus
   - Balance between bitwidth (model size) and accuracy
   - Look for the lowest bitwidth with acceptable accuracy

### Example Interpretations

**Scenario 1: Strong Positive Correlation**
```
Bitwidth 8:  Accuracy = 0.85 ± 0.02
Bitwidth 16: Accuracy = 0.91 ± 0.015
Bitwidth 32: Accuracy = 0.94 ± 0.01
```
→ Higher precision needed for this task

**Scenario 2: Plateau Effect**
```
Bitwidth 8:  Accuracy = 0.85 ± 0.03
Bitwidth 16: Accuracy = 0.92 ± 0.02
Bitwidth 32: Accuracy = 0.92 ± 0.015
```
→ 16-bit is optimal (no benefit from 32-bit)

**Scenario 3: High Variance**
```
Bitwidth 8:  Accuracy = 0.80 ± 0.08 (large variance!)
Bitwidth 16: Accuracy = 0.88 ± 0.02
Bitwidth 32: Accuracy = 0.90 ± 0.015
```
→ Low bitwidths are unstable for this model

## Files Modified

1. **`src2/train_test/quantization_train_test_utils.py`**
   - Added `get_average_bitwidth()` function
   - Enhanced `validate_random_bitwidths()` to track bitwidths
   - Added `plot_bitwidth_vs_accuracy()` visualization function
   - Updated training loop to log bitwidth metrics
   - Added bitwidth vs accuracy plot logging

## Technical Details

### How Bitwidth is Calculated

```python
# For each quantized Conv layer:
1. Check if layer has 'curr_bitwidth' attribute
2. Extract current bitwidth value
3. Calculate mean across all layers
```

### Plot Generation

```python
# Creates bins for unique bitwidth values
1. Round bitwidths to 1 decimal place
2. Group accuracies by bitwidth bin
3. Calculate statistics (mean, std, min, max) per bin
4. Plot scatter + mean line + error bars
```

### Performance Impact

- Minimal: Only calculates during validation (not training)
- Plot generation: ~0.1-0.2 seconds every 5 epochs
- No impact on training speed

## Tips

1. **Number of Configs**: Use at least 4-5 random configs for reliable statistics
2. **Bitwidth Range**: Test a wide range (e.g., [2, 4, 8, 16, 32]) to see trends
3. **Plot Frequency**: Adjust plot generation frequency if storage is a concern
4. **TensorBoard**: Compare multiple experiments side-by-side using TensorBoard's comparison feature

## Example TensorBoard View

### Scalar Metrics
```
Bitwidth/mean_bitwidth     → Shows average bitwidth over time
Bitwidth/std_bitwidth      → Shows bitwidth variance over time
Validation/mean_acc        → Shows accuracy improving with training
```

### Image Plots
```
Bitwidth_Analysis/bitwidth_vs_accuracy → Scatter plot every 5 epochs
```

## Related Documentation

- `QUANTIZATION_QUICK_START.md` - Quantization training basics
- `EXPERIMENT_TRACKING_README.md` - Experiment management
- `src2/train_test/quantization_train_test_utils.py` - Implementation details

## Summary

These features provide comprehensive insights into how bitwidth affects model accuracy, enabling you to:
- Find optimal bitwidth configurations
- Understand bitwidth-accuracy tradeoffs
- Detect training instabilities
- Make informed decisions about model quantization

All logging is automatic when using `random_bitwidths` validation!

