# TensorBoard Graphs Reference

Complete documentation of all TensorBoard graphs displayed across different quantization schemes and training methods.

## Table of Contents
1. [Common Graphs (All Schemes)](#common-graphs-all-schemes)
2. [Training Method Specific Graphs](#training-method-specific-graphs)
3. [Validation Function Specific Graphs](#validation-function-specific-graphs)
4. [Quantization Method Specific Graphs](#quantization-method-specific-graphs)
5. [Final Epoch Analysis Graphs](#final-epoch-analysis-graphs)
6. [Per-Layer Analysis Graphs](#per-layer-analysis-graphs)
7. [Batch-Level Tracking Graphs](#batch-level-tracking-graphs)
8. [Method Comparison Graphs](#method-comparison-graphs)

---

## Common Graphs (All Schemes)

These graphs are logged for **ALL** training methods and quantization schemes:

### Epoch-Level Metrics
- **`Loss/train`** - Training loss per epoch
- **`Loss/val`** - Validation loss per epoch
- **`Accuracy/train`** - Training accuracy per epoch
- **`Accuracy/val`** - Validation accuracy per epoch
- **`Learning_Rate`** - Learning rate per epoch

---

## Training Method Specific Graphs

### 1. Vanilla Single Precision Training
**Training Method:** `training_method: "vanilla_single_precision_training"`

#### Batch-Level (Logged every 50 batches):
- **`Train/Batch_Loss`** - Training loss per batch (global_step)

**Note:** This is the simplest training method with minimal logging.

---

### 2. Joint Quantization
**Training Method:** `training_method: "joint_quantization"`

#### Batch-Level (Logged every 50 batches):
- **`Train/Batch_Loss`** - Average training loss per batch (after averaging multiple forward passes)
- **`Train/RobustQuant_Reg`** - RobustQuant regularization term (only if RobustQuant is enabled)

**Characteristics:**
- Multiple forward passes per batch with random bitwidths
- Losses are averaged before backward pass
- Minimal batch-level logging

---

### 3. Sandwich Quantization
**Training Method:** `training_method: "sandwich_quantization"`

#### Batch-Level (Logged every 50 batches):
- **`Train/Batch_Loss`** - Total weighted loss: `loss_min + λ₁*loss_max + λ₂*loss_random`
- **`Train/Loss_Min`** - Loss from MIN bitwidth forward pass
- **`Train/Loss_Max`** - Loss from MAX bitwidth forward pass
- **`Train/Loss_Random`** - Average loss from random configuration forward passes
- **`Train/RobustQuant_Reg`** - RobustQuant regularization term (only if RobustQuant is enabled)

**Characteristics:**
- Always includes MIN and MAX bitwidth forward passes
- Includes N random configuration forward passes
- Detailed loss component tracking

---

## Validation Function Specific Graphs

### 1. Simple Validation
**Validation Function:** `validation_function: "simple_validation"`

**Graphs:**
- None beyond common epoch-level graphs (`Loss/val`, `Accuracy/val`)

**Characteristics:**
- Single fixed bitwidth validation
- Minimal logging

---

### 2. Random Bitwidths Validation
**Validation Function:** `validation_function: "random_bitwidths"`

#### Overall Statistics (Per Epoch):
- **`Validation/mean_acc`** - Mean accuracy across all random configurations
- **`Validation/std_acc`** - Standard deviation of accuracy
- **`Validation/min_acc`** - Minimum accuracy observed
- **`Validation/max_acc`** - Maximum accuracy observed

#### Per-Bitwidth Statistics (Per Epoch):
For each bitwidth option (e.g., 4, 6, 8, 16):
- **`Validation/Bitwidth_{X}/mean_acc`** - Mean accuracy for configurations closest to X-bit
- **`Validation/Bitwidth_{X}/std_acc`** - Standard deviation for X-bit
- **`Validation/Bitwidth_{X}/min_acc`** - Minimum accuracy for X-bit
- **`Validation/Bitwidth_{X}/max_acc`** - Maximum accuracy for X-bit

#### Fixed Bitwidth Tests (Per Epoch):
For each bitwidth option in `bitwidth_options`:
- **`Validation/Fixed_{X}bit/accuracy`** - Accuracy when ALL layers use X-bit
- **`Validation/Fixed_{X}bit/loss`** - Loss when ALL layers use X-bit

**Example:** If `bitwidth_options: [4, 8, 16]`, you get:
- `Validation/Fixed_4bit/accuracy`
- `Validation/Fixed_4bit/loss`
- `Validation/Fixed_8bit/accuracy`
- `Validation/Fixed_8bit/loss`
- `Validation/Fixed_16bit/accuracy`
- `Validation/Fixed_16bit/loss`

#### Candlestick Plots (Per Epoch):
- **`Validation/Candlestick_Accuracy`** - Candlestick plot showing accuracy distribution by bitwidth
  - Red line: Mean
  - Blue box: Mean ± Std
  - Black whiskers: Min to Max range
- **`Validation/Candlestick_Loss`** - Candlestick plot showing loss distribution by bitwidth
  - Same structure as accuracy plot

**Characteristics:**
- Tests multiple random bitwidth configurations
- Tracks statistics per bitwidth
- Provides visual candlestick plots
- Tests fixed bitwidths separately

---

## Quantization Method Specific Graphs

### RobustQuant
**Quantization Method:** `quantization_method: "robustquant"`

#### Additional Graphs:
- **`Train/RobustQuant_Reg`** - RobustQuant regularization term (kurtosis-based)
  - Logged during training (batch-level for joint/sandwich, not for vanilla)
  - Shows weight distribution regularization strength

**When Active:**
- Only when `quantization_method == "robustquant"` OR `wdr_lambda > 0`
- Appears in both joint quantization and sandwich quantization training

---

## Final Epoch Analysis Graphs

**When:** Only at the **final epoch** (after training completes)

### Average Bitwidth vs Accuracy Analysis
- **`Validation/AvgBitwidth_vs_Accuracy`** - Two-panel plot:
  - **Left Panel:** Accuracy vs Average Bitwidth with error bars (mean ± std) and min-max range shading
  - **Right Panel:** Scatter plot with std error bars showing accuracy distribution

**Purpose:**
- Verify convergence hypothesis
- Check if accuracy varies with average bitwidth
- Uses diverse sampling to get many different average bitwidths

**Characteristics:**
- Tests diverse average bitwidth configurations (targeted sampling)
- Creates comprehensive plot showing relationship between average bitwidth and accuracy
- Helps identify if model maintains performance across bitwidths

---

## Complete Graph Summary by Scheme

### Scheme 1: Vanilla Single Precision + Simple Validation
**Config:**
```yaml
training_method: "vanilla_single_precision_training"
validation_function: "simple_validation"
```

**Graphs:**
- `Loss/train` (epoch)
- `Loss/val` (epoch)
- `Accuracy/train` (epoch)
- `Accuracy/val` (epoch)
- `Learning_Rate` (epoch)
- `Train/Batch_Loss` (batch, every 50)

**Total: 6 graphs**

---

### Scheme 2: Joint Quantization + Random Bitwidths (Any Precision)
**Config:**
```yaml
training_method: "joint_quantization"
validation_function: "random_bitwidths"
quantization_method: "any_precision"
```

**Graphs:**
- `Loss/train` (epoch)
- `Loss/val` (epoch)
- `Accuracy/train` (epoch)
- `Accuracy/val` (epoch)
- `Learning_Rate` (epoch)
- `Train/Batch_Loss` (batch, every 50)
- `Validation/mean_acc` (epoch)
- `Validation/std_acc` (epoch)
- `Validation/min_acc` (epoch)
- `Validation/max_acc` (epoch)
- `Validation/Bitwidth_{X}/mean_acc` (epoch, per bitwidth)
- `Validation/Bitwidth_{X}/std_acc` (epoch, per bitwidth)
- `Validation/Bitwidth_{X}/min_acc` (epoch, per bitwidth)
- `Validation/Bitwidth_{X}/max_acc` (epoch, per bitwidth)
- `Validation/Fixed_{X}bit/accuracy` (epoch, per bitwidth)
- `Validation/Fixed_{X}bit/loss` (epoch, per bitwidth)
- `Validation/Candlestick_Accuracy` (epoch, figure)
- `Validation/Candlestick_Loss` (epoch, figure)
- `Validation/AvgBitwidth_vs_Accuracy` (final epoch only, figure)

**Total: ~20+ scalar graphs + 3 figure plots**

---

### Scheme 3: Joint Quantization + Random Bitwidths (RobustQuant)
**Config:**
```yaml
training_method: "joint_quantization"
validation_function: "random_bitwidths"
quantization_method: "robustquant"
```

**Graphs:**
- All graphs from Scheme 2, PLUS:
- `Train/RobustQuant_Reg` (batch, every 50)

**Total: ~21+ scalar graphs + 3 figure plots**

---

### Scheme 4: Sandwich Quantization + Random Bitwidths
**Config:**
```yaml
training_method: "sandwich_quantization"
validation_function: "random_bitwidths"
```

**Graphs:**
- `Loss/train` (epoch)
- `Loss/val` (epoch)
- `Accuracy/train` (epoch)
- `Accuracy/val` (epoch)
- `Learning_Rate` (epoch)
- `Train/Batch_Loss` (batch, every 50)
- `Train/Loss_Min` (batch, every 50)
- `Train/Loss_Max` (batch, every 50)
- `Train/Loss_Random` (batch, every 50)
- `Train/RobustQuant_Reg` (batch, every 50, if RobustQuant enabled)
- All validation graphs from Scheme 2

**Total: ~24+ scalar graphs + 3 figure plots**

---

### Scheme 5: Sandwich Quantization + Random Bitwidths (RobustQuant)
**Config:**
```yaml
training_method: "sandwich_quantization"
validation_function: "random_bitwidths"
quantization_method: "robustquant"
```

**Graphs:**
- All graphs from Scheme 4 (includes RobustQuant regularization)

**Total: ~25+ scalar graphs + 3 figure plots**

---

### Scheme 6: Joint Quantization + Random Bitwidths (AdaBits)
**Config:**
```yaml
training_method: "joint_quantization"
validation_function: "random_bitwidths"
quantization_method: "adabits"
```

**Graphs:**
- Same as Scheme 2 (no additional graphs, AdaBits is transparent to logging)

**Total: ~20+ scalar graphs + 3 figure plots**

---

## Graph Categories in TensorBoard

### Scalars Tab
All `writer.add_scalar()` calls appear here:
- Training metrics (batch and epoch level)
- Validation metrics (epoch level)
- Per-bitwidth statistics
- Fixed bitwidth tests
- Learning rate
- Regularization terms

### Images Tab
All `writer.add_figure()` calls appear here:
- `Validation/Candlestick_Accuracy` - Updated every epoch
- `Validation/Candlestick_Loss` - Updated every epoch
- `Validation/AvgBitwidth_vs_Accuracy` - Only at final epoch

---

## Key Insights from Graphs

### Convergence Detection
- **Narrow candlesticks** = Convergence (all bitwidths perform similarly)
- **Wide candlesticks** = Diversity maintained (bitwidths perform differently)
- **Compare:** Joint quantization vs Sandwich quantization candlestick widths

### 4-bit Drop Hypothesis
- **Compare:** `Validation/Fixed_4bit/accuracy` vs `Validation/Fixed_8bit/accuracy`
- **Check:** If 4-bit accuracy is significantly lower than 8-bit

### Method Comparison
- **Any Precision:** Baseline joint quantization
- **RobustQuant:** Same as any_precision + regularization term
- **AdaBits:** Same as any_precision (weight normalization is transparent)
- **Sandwich:** More detailed loss component tracking

### Average Bitwidth Analysis
- **Final epoch plot:** Shows if accuracy varies with average bitwidth
- **Flat line** = Convergence (all average bitwidths perform similarly)
- **Sloping line** = Diversity (higher average bitwidth = better accuracy)

---

## Example TensorBoard Navigation

1. **Open TensorBoard:**
   ```bash
   tensorboard --logdir=experiments/YOUR_EXPERIMENT/tensorboard
   ```

2. **Scalars Tab:**
   - Navigate to `Validation/` folder to see all validation metrics
   - Navigate to `Train/` folder to see training metrics
   - Use filter to find specific graphs (e.g., "Fixed_4bit")

3. **Images Tab:**
   - View candlestick plots updated every epoch
   - View final epoch average bitwidth analysis

4. **Compare Runs:**
   - Use TensorBoard's run selector to compare different quantization methods
   - Overlay graphs to see differences

---

## Per-Layer Analysis Graphs

### Per-Layer Bitwidth Distribution
**When:** Every epoch (with `random_bitwidths` validation)

- **`Validation/PerLayer_Bitwidth_Distribution`** - Histogram showing bitwidth selection frequency for each layer
  - Subplot for each quantized layer
  - Shows how often each bitwidth is selected during random sampling
  - Helps identify if certain layers prefer specific bitwidths

**Purpose:**
- Track which bitwidths each layer receives during random sampling
- Identify layer-specific bitwidth preferences
- Verify random sampling is working correctly

---

### Per-Layer Sensitivity Analysis
**When:** Every 5 epochs (to reduce computation) + Final epoch

#### Scalar Metrics (Per Layer):
- **`Sensitivity/Layer_{X}/sensitivity`** - Sensitivity score for layer X
- **`Sensitivity/Layer_{X}/loss_change`** - Loss change when layer X uses min bitwidth

#### Figure Plot:
- **`Validation/PerLayer_Sensitivity`** - Two-panel plot:
  - **Left Panel:** Sensitivity scores (bar chart)
  - **Right Panel:** Loss change (bar chart, red=increase, green=decrease)
- **`Validation/Final_PerLayer_Sensitivity`** - Final epoch sensitivity analysis

**Purpose:**
- Identify which layers are most sensitive to bitwidth changes
- Understand layer importance for quantization
- Guide bitwidth allocation strategies

**Method:**
- Baseline: All layers at max bitwidth
- Test: Change one layer to min bitwidth, measure loss change
- Sensitivity = |loss_change| / baseline_loss

---

## Batch-Level Tracking Graphs

### Fixed Bitwidth Accuracy Over Time
**When:** Every 50 batches during training

- **`Train/Fixed_{X}bit_BatchAccuracy`** - Batch-level accuracy when ALL layers use X-bit
  - Tracked for each bitwidth in `bitwidth_options`
  - Shows convergence over training time (not just epoch-level)
  - Helps identify when different bitwidths start converging

**Purpose:**
- Track accuracy at fixed bitwidths throughout training (not just validation)
- See convergence patterns in real-time
- Compare 4-bit vs 8-bit vs 16-bit accuracy evolution

**Available In:**
- Joint Quantization training
- Sandwich Quantization training

**Note:** This adds computational overhead (extra forward passes), but provides detailed tracking.

---

## Method Comparison Graphs

### Method Comparison Plot
**When:** Final epoch (per method) + Can be generated separately

- **`Comparison/Method_Comparison`** - Four-panel comparison plot:
  - **Top Left:** Overall accuracy comparison (bar chart with error bars)
  - **Top Right:** Accuracy range comparison (min/mean/max bars)
  - **Bottom:** Per-bitwidth accuracy comparison (line plot across methods)

**Purpose:**
- Compare different quantization methods side-by-side
- Visualize which method maintains best accuracy across bitwidths
- Identify method-specific patterns

**Usage:**
- Automatically generated at final epoch for each method
- Can be used with `compare_methods.py` script to compare multiple experiments

---

## Summary

**Total Graph Types:**
- **Scalar Graphs:** ~30+ unique metrics
- **Figure Plots:** 7 types:
  1. Candlestick Accuracy
  2. Candlestick Loss
  3. Average Bitwidth vs Accuracy
  4. Per-Layer Bitwidth Distribution
  5. Per-Layer Sensitivity Analysis
  6. Final Per-Layer Sensitivity
  7. Method Comparison

**Graph Frequency:**
- **Epoch-level:** Most validation and training metrics
- **Batch-level:** Training losses + Fixed bitwidth accuracy (every 50 batches)
- **Every 5 epochs:** Per-layer sensitivity analysis
- **Final epoch:** Average bitwidth analysis, final sensitivity, method comparison

**Most Comprehensive Logging:**
- **Sandwich Quantization + Random Bitwidths Validation** provides the most detailed tracking
- Includes: batch-level fixed bitwidth tracking, per-layer analysis, sensitivity analysis

