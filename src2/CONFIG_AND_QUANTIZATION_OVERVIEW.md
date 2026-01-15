# Configuration and Quantization System Overview

## Table of Contents
1. [Configuration Structure](#configuration-structure)
2. [DoReFa Quantization Setup](#dorefa-quantization-setup)
3. [Joint Quantization Training](#joint-quantization-training)
4. [How They Work Together](#how-they-work-together)
5. [Execution Flow](#execution-flow)
6. [Key Files and Functions](#key-files-and-functions)

---

## Configuration Structure

### YAML Configuration Files

The quantization system is configured through YAML files (e.g., `Parkland.yaml`, `PAMAP2.yaml`). The configuration has a hierarchical structure:

```yaml
quantization:
  enable: True                    # Master switch for quantization
  Conv: "QuanConv"               # Use QuanConv instead of standard Conv2d
  
  # Method-specific configurations
  dorefa:
    bitwidth_options: [8, 16, 32]              # Available bitwidths
    weight_quantization: "dorefa"              # Weight quantization method
    activation_quantization: "dorefa"          # Activation quantization method
    training_method: "joint_quantization"      # Training strategy
    validation_function: "random_bitwidths"    # Validation strategy
    switchable_clipping: True                  # For PACT (future use)
    sat_weight_normalization: False            # Weight normalization
  
  # Joint quantization settings (shared across methods)
  joint_quantization:
    joint_quantization_batch_size: 2          # Forward passes per batch
  
  # Random bitwidths validation settings (top-level, not nested under method)
  random_bitwidths:
    num_bitwidths: 3                           # Number of random configs to test during validation
```

### Command-Line Arguments

The quantization method is specified via command-line:

```bash
python train.py \
  --model ResNet \
  --model_variant resnet18 \
  --yaml_path data/Parkland.yaml \
  --quantization_method dorefa \
  --gpu 0
```

The `--quantization_method` argument selects which nested config to use (e.g., `dorefa`, `any_precision`).

---

## DoReFa Quantization Setup

### 1. Model Creation (`models/create_models.py`)

When quantization is enabled:
- Standard `Conv2d` layers are replaced with `QuanConv` layers
- The model is created with `Conv=QuanConv` parameter

```python
# In create_models.py
if quantization_enabled:
    from models.QuantModules import QuanConv
    Conv = QuanConv
else:
    Conv = None  # Uses standard Conv2d
```

### 2. Quantization Layer Setup (`models/QuantModules.py`)

Each `QuanConv` layer:
- Stores quantization configuration
- Has methods to set bitwidth dynamically
- Supports DoReFa weight and activation quantization

**Key Components:**

#### DoReFa Weight Quantization (`DoReFaW`)
```python
# Process: tanh → normalize → quantize → dequantize
w = torch.tanh(inp)
w = w / (2 * maxv) + 0.5
w = 2 * quantize(w, nbit_w) - 1
```

#### DoReFa Activation Quantization (`DoReFaA`)
```python
# Process: clamp → quantize
return quantize(torch.clamp(inp, 0, 1), nbit_a)
```

### 3. Configuration Loading (`quantization_train_test_utils.py`)

The `setup_quantization_layers()` function:
1. Extracts the quantization method config (e.g., `config['quantization']['dorefa']`)
2. Calls `setup_quantize_funcs()` on each `QuanConv` layer
3. Configures weight and activation quantizers based on config

```python
def setup_quantization_layers(model, config, quant_config):
    # For each QuanConv layer:
    module.setup_quantize_funcs(args)
```

**Inside `setup_quantize_funcs()`:**
- Reads `weight_quantization` and `activation_quantization` from config
- Creates `DoReFaW()` for weights and `DoReFaA()` for activations
- Stores `bitwidth_options` for dynamic bitwidth selection
- Sets up PACT alpha if needed (not used for DoReFa)

---

## Joint Quantization Training

### Concept

Joint quantization trains the model to work with **multiple bitwidth configurations simultaneously**. For each batch:
1. Multiple forward passes are performed with **different random bitwidth configurations**
2. Losses from all passes are **averaged**
3. A single backward pass updates the model with the averaged gradient

### Implementation (`train_epoch_joint_quantization()`)

```python
# For each batch:
for _ in range(joint_quantization_batch_size):  # Default: 2 passes
    # 1. Set random bitwidths for all layers
    set_random_bitwidth_all_layers(model, bitwidth_options)
    
    # 2. Forward pass
    outputs = model(data)
    loss = loss_fn(outputs, labels)
    accumulated_loss += loss

# 3. Average the loss
avg_loss = accumulated_loss / joint_quantization_batch_size

# 4. Single backward pass
avg_loss.backward()
optimizer.step()
```

### Random Bitwidth Allocation

The `set_random_bitwidth_all_layers()` function:
- Iterates through all `QuanConv` layers in the model
- For each layer, **independently** samples a bitwidth from `bitwidth_options`
- Calls `layer.set_bitwidth(bw)` to set the bitwidth

**Example:**
- Layer 1 might get bitwidth=8
- Layer 2 might get bitwidth=16
- Layer 3 might get bitwidth=8
- etc.

This creates **mixed-precision** configurations where different layers use different bitwidths.

---

## How They Work Together

### Training Flow

```
1. CONFIG LOADING
   ├─ YAML file loaded (parse_args_utils.py)
   ├─ Command-line args merged (quantization_method)
   └─ Full config dictionary created

2. MODEL CREATION
   ├─ Check: quantization.enable == True?
   ├─ If yes: Use QuanConv instead of Conv2d
   └─ Model created with QuanConv layers (create_models.py)

3. QUANTIZATION SETUP
   ├─ Extract method config: config['quantization']['dorefa']
   ├─ For each QuanConv layer:
   │   ├─ setup_quantize_funcs() called
   │   ├─ DoReFaW() created for weights
   │   ├─ DoReFaA() created for activations
   │   └─ bitwidth_options stored
   └─ Model ready for training

4. TRAINING LOOP (train_with_quantization)
   ├─ For each epoch:
   │   ├─ Training Phase (train_epoch_joint_quantization)
   │   │   ├─ For each batch:
   │   │   │   ├─ For joint_quantization_batch_size times:
   │   │   │   │   ├─ set_random_bitwidth_all_layers()
   │   │   │   │   ├─ Forward pass (DoReFa quantization applied)
   │   │   │   │   └─ Accumulate loss
   │   │   │   ├─ Average losses
   │   │   │   └─ Backward pass (single gradient update)
   │   │   └─ End batch
   │   │
   │   └─ Validation Phase (validate_random_bitwidths)
   │       ├─ Read num_bitwidths from config['random_bitwidths']
   │       ├─ For num_bitwidths times:
   │       │   ├─ set_random_bitwidth_all_layers()
   │       │   ├─ Forward pass on validation set
   │       │   └─ Record accuracy/loss
   │       └─ Report statistics (mean, min, max, std)
   └─ End epoch
```

### Key Interactions

1. **DoReFa provides the quantization functions**: `DoReFaW` and `DoReFaA` are the actual quantization implementations that quantize weights and activations.

2. **Joint quantization uses DoReFa with random bitwidths**: Each forward pass in joint quantization:
   - Sets random bitwidths per layer
   - Uses DoReFa quantizers with those bitwidths
   - Accumulates gradients across multiple configurations

3. **Configuration drives everything**: The YAML config specifies:
   - Which quantization method (DoReFa)
   - Which training method (joint_quantization)
   - Which validation method (random_bitwidths)
   - Available bitwidths and batch sizes

---

## Execution Flow

### Entry Point: `train.py`

```python
def main():
    # 1. Load config
    config = get_config()  # Merges YAML + command-line args
    
    # 2. Create model
    model = create_model(config)  # Uses QuanConv if quantization enabled
    
    # 3. Check quantization flag
    if config.get("quantization", {}).get("enable", False):
        # 4. Use quantization training
        train_with_quantization(...)
    else:
        # Standard training
        train(...)
```

### Quantization Training: `train_with_quantization()`

```python
def train_with_quantization(...):
    # 1. Extract quantization method config
    quantization_method = config.get('quantization_method', 'dorefa')
    quant_config = config['quantization'][quantization_method]
    
    # 2. Setup quantization layers
    model = setup_quantization_layers(model, config, quant_config)
    
    # 3. Extract training/validation methods
    training_method = quant_config.get('training_method', 'joint_quantization')
    validation_function = quant_config.get('validation_function', 'random_bitwidths')
    
    # 4. Training loop
    for epoch in range(num_epochs):
        if training_method == "joint_quantization":
            train_epoch_joint_quantization(...)
        
        if validation_function == "random_bitwidths":
            # Get num_bitwidths from top-level config['random_bitwidths']
            # Get bitwidth_options from quant_config
            validate_random_bitwidths(...)
```

### Forward Pass with DoReFa

When `model(data)` is called:

```python
# In QuanConv.forward()
# 1. Quantize weights using DoReFaW
w_quantized = self.quantize_w(self.weight, nbit_w=self.curr_bitwidth)

# 2. Standard convolution
out = F.conv2d(inp, w_quantized, ...)

# 3. Quantize activations using DoReFaA
out = self.quantize_a(out, nbit_a=self.curr_bitwidth)

# 4. Return quantized output
return out
```

---

## Key Files and Functions

### Configuration Files
- **`data/*.yaml`**: Dataset and quantization configuration files
- **`dataset_utils/parse_args_utils.py`**: 
  - `parse_args()`: Parse command-line arguments
  - `load_yaml_config()`: Load YAML file
  - `get_config()`: Merge args + YAML into single config dict

### Model Creation
- **`models/create_models.py`**:
  - `create_model()`: Creates model with QuanConv if quantization enabled

### Quantization Implementation
- **`models/QuantModules.py`**:
  - `DoReFaW`: DoReFa weight quantization
  - `DoReFaA`: DoReFa activation quantization
  - `QuanConv`: Quantized convolution layer
    - `setup_quantize_funcs()`: Configure quantizers from config
    - `set_bitwidth()`: Set current bitwidth for layer
    - `forward()`: Forward pass with quantization

### Training
- **`train_test/train.py`**:
  - `main()`: Entry point, routes to quantization or standard training

- **`train_test/quantization_train_test_utils.py`**:
  - `setup_quantization_layers()`: Setup all QuanConv layers
  - `set_random_bitwidth_all_layers()`: Set random bitwidths per layer
  - `train_epoch_joint_quantization()`: Joint quantization training loop
  - `validate_random_bitwidths()`: Validation with multiple bitwidth configs
  - `train_with_quantization()`: Main quantization training orchestrator

---

## Configuration Examples

### Example 1: DoReFa with Joint Quantization (Current Parkland.yaml)

```yaml
quantization:
  enable: True
  Conv: "QuanConv"
  
  dorefa:
    bitwidth_options: [8, 16, 32]
    weight_quantization: "dorefa"
    activation_quantization: "dorefa"
    training_method: "joint_quantization"
    validation_function: "random_bitwidths"
    switchable_clipping: True
    sat_weight_normalization: False
  
  joint_quantization:
    joint_quantization_batch_size: 2
  
  random_bitwidths:
    num_bitwidths: 3
```

**What this does:**
- Uses DoReFa for both weights and activations
- Trains with 2 forward passes per batch (joint quantization)
- Each forward pass uses random bitwidths from [8, 16, 32]
- Validates with 3 random bitwidth configurations

### Example 2: Vanilla Single Precision (Alternative)

```yaml
dorefa:
  bitwidth_options: [8]
  weight_quantization: "dorefa"
  activation_quantization: "dorefa"
  training_method: "vanilla_single_precision_training"
  validation_function: "simple_validation"
```

**What this does:**
- Uses DoReFa with fixed 8-bit quantization
- No joint quantization (single forward pass per batch)
- Simple validation with fixed 8-bit

### Example 3: Mixed Methods

```yaml
dorefa:
  bitwidth_options: [4, 6, 8]
  weight_quantization: "dorefa"
  activation_quantization: "pact"  # Mix DoReFa weights with PACT activations
  training_method: "joint_quantization"
  validation_function: "random_bitwidths"
```

**What this does:**
- DoReFa for weights, PACT for activations
- Joint quantization with bitwidths [4, 6, 8]
- Random bitwidths validation

---

## Summary

1. **Configuration**: YAML files define quantization settings, command-line selects method
2. **DoReFa**: Provides weight (`DoReFaW`) and activation (`DoReFaA`) quantization functions
3. **Joint Quantization**: Trains with multiple random bitwidth configs per batch, averages losses
4. **Integration**: DoReFa quantizers are used within joint quantization training, with bitwidths set randomly per layer per forward pass
5. **Flow**: Config → Model Creation → Quantization Setup → Training Loop → Validation

The system is designed to be flexible: you can use DoReFa with different training strategies (joint quantization, vanilla), different validation strategies (random bitwidths, simple), and mix quantization methods (DoReFa weights + PACT activations).

