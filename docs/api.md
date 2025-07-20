# API Reference

This document provides detailed API documentation for all classes and functions in the Clippy AdaGrad implementation.

## Table of Contents

- [ClippyAdagrad Optimizer](#clippyadagrad-optimizer)
- [Model Architectures](#model-architectures)
- [Dataset Classes](#dataset-classes)
- [Utility Functions](#utility-functions)

## ClippyAdagrad Optimizer

### `ClippyAdagrad`

```python
class ClippyAdagrad(Optimizer):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, 
                 initial_accumulator_value=0, eps=1e-10, foreach=None, 
                 maximize=False, differentiable=False, lambda_rel=0.5, lambda_abs=1e-2):
```

**Description**: Implements AdaGrad algorithm with Clippy gradient clipping.

**Parameters**:
- `params` (iterable): Parameters to optimize
- `lr` (float, optional): Learning rate (default: 1e-2)
- `lr_decay` (float, optional): Learning rate decay (default: 0)
- `weight_decay` (float, optional): Weight decay (L2 penalty) (default: 0)
- `eps` (float, optional): Term added to denominator for numerical stability (default: 1e-10)
- `lambda_rel` (float, optional): Relative clipping threshold (default: 0.5)
- `lambda_abs` (float, optional): Absolute clipping threshold (default: 1e-2)

**Methods**:
- `step(closure=None)`: Performs a single optimization step

**Example**:
```python
optimizer = ClippyAdagrad(model.parameters(), lr=1e-1, lambda_rel=0.5, lambda_abs=1e-2)
```

## Model Architectures

### `SharedBottomModel`

```python
class SharedBottomModel(nn.Module):
    def __init__(self, field_dims, numerical_num, embed_dim=128, 
                 bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), 
                 task_num=2, dropout=0.2):
```

**Description**: Shared Bottom multitask learning model with task-specific towers.

**Parameters**:
- `field_dims` (list): Dimensions of categorical fields
- `numerical_num` (int): Number of numerical features
- `embed_dim` (int): Embedding dimension (default: 128)
- `bottom_mlp_dims` (tuple): Shared bottom MLP dimensions (default: (512, 256))
- `tower_mlp_dims` (tuple): Task-specific tower MLP dimensions (default: (128, 64))
- `task_num` (int): Number of tasks (default: 2)
- `dropout` (float): Dropout rate (default: 0.2)

**Forward Pass**:
```python
def forward(self, categorical_fields, numerical_fields):
    # Returns: List of task predictions
```

### `NCF`

```python
class NCF(nn.Module):
    def __init__(self, field_dims, numerical_num, embed_dim=128, 
                 bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), 
                 task_num=2, dropout=0.2):
```

**Description**: Neural Collaborative Filtering model.

**Parameters**: Same as SharedBottomModel

### `LogisticRegressionModel`

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, task_num=2):
```

**Description**: Logistic Regression baseline model.

**Parameters**:
- `input_dim` (int): Total input dimension (one-hot encoded)
- `task_num` (int): Number of tasks (default: 2)

## Dataset Classes

### `AliExpressDataset`

```python
class AliExpressDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, n_rows=None):
```

**Description**: AliExpress e-commerce dataset loader.

**Parameters**:
- `dataset_path` (str): Path to CSV dataset file
- `n_rows` (int, optional): Number of rows to load (default: None)

**Attributes**:
- `categorical_data`: Categorical features (16 fields)
- `numerical_data`: Numerical features (6 fields)
- `labels`: Target labels (2 tasks)
- `field_dims`: Dimensions of categorical fields
- `numerical_num`: Number of numerical features

**Methods**:
- `__len__()`: Returns dataset size
- `__getitem__(index)`: Returns (categorical_fields, numerical_fields, labels)

## Utility Functions

### `clippy_adagrad`

```python
def clippy_adagrad(params, grads, state_sums, state_steps, lr, weight_decay, 
                   lr_decay, eps, has_sparse_grad=None, foreach=None, 
                   differentiable=False, maximize=False, lambda_rel=0.5, lambda_abs=1e-2):
```

**Description**: Functional API for Clippy AdaGrad algorithm.

**Parameters**:
- `params` (List[Tensor]): Parameters to optimize
- `grads` (List[Tensor]): Gradients
- `state_sums` (List[Tensor]): State sums for AdaGrad
- `state_steps` (List[Tensor]): State steps
- `lambda_rel` (float): Relative clipping threshold
- `lambda_abs` (float): Absolute clipping threshold

### `_compute_clippy_factor`

```python
def _compute_clippy_factor(param, grad, std, clr, lambda_rel, lambda_abs, foreach=True):
```

**Description**: Computes Clippy gradient clipping factor.

**Parameters**:
- `param` (Tensor): Parameter tensor
- `grad` (Tensor): Gradient tensor
- `std` (float): Gradient standard deviation
- `clr` (float): Current learning rate
- `lambda_rel` (float): Relative threshold
- `lambda_abs` (float): Absolute threshold

**Returns**: Clipping factor tensor

## Training Functions

### `train_model`

```python
def train_model(model, train_loader, optimizer, criterion, device, log_interval=100):
```

**Description**: Training loop for multitask models.

**Parameters**:
- `model`: PyTorch model
- `train_loader`: DataLoader for training data
- `optimizer`: Optimizer instance
- `criterion`: Loss function
- `device`: Device to train on
- `log_interval`: Logging interval

### `evaluate_model`

```python
def evaluate_model(model, test_loader, criterion, device, task_num=2):
```

**Description**: Evaluation function for multitask models.

**Parameters**:
- `model`: PyTorch model
- `test_loader`: DataLoader for test data
- `criterion`: Loss function
- `device`: Device to evaluate on
- `task_num`: Number of tasks

**Returns**: Dictionary with AUC and loss for each task

## Visualization Functions

### `plot_training_losses`

```python
def plot_training_losses(results_files):
```

**Description**: Plots training loss curves from results files.

**Parameters**:
- `results_files` (list): List of result file paths

### `plot_test_metrics`

```python
def plot_test_metrics(results_files):
```

**Description**: Plots test AUC and loss comparisons.

**Parameters**:
- `results_files` (list): List of result file paths

## Configuration

### Default Hyperparameters

```python
# Training Configuration
batch_size = 2048
embed_dim = 128
learning_rate = 1e-1
task_num = 2

# Clippy Parameters
lambda_rel = 0.5
lambda_abs = 1e-2

# Model Architecture
bottom_mlp_dims = (512, 256)
tower_mlp_dims = (128, 64)
dropout = 0.2
```

## Error Handling

The implementation includes comprehensive error handling for:

- Invalid parameter values
- Gradient computation errors
- Memory issues
- Device compatibility

## Performance Notes

- **Memory Usage**: Models are optimized for GPU memory efficiency
- **Gradient Clipping**: Clippy method improves training stability
- **Multitask Learning**: Efficient shared parameter updates
- **Batch Processing**: Optimized for large batch sizes (2048)

## Examples

### Basic Training Example

```python
import torch
from clippyadagrad import ClippyAdagrad
from models.sharedbottom import SharedBottomModel
from aliexpress import AliExpressDataset

# Initialize model
model = SharedBottomModel(field_dims, numerical_num)
optimizer = ClippyAdagrad(model.parameters(), lr=1e-1)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Custom Clippy Parameters

```python
# Custom Clippy configuration
optimizer = ClippyAdagrad(
    model.parameters(),
    lr=1e-1,
    lambda_rel=0.3,  # More aggressive relative clipping
    lambda_abs=5e-3  # Lower absolute threshold
)
``` 