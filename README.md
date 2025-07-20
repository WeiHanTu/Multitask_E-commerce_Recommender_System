# Clippy AdaGrad for Multitask Learning in Recommender Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a comprehensive implementation of **Clippy AdaGrad** optimizer for multitask learning in recommender systems, based on the paper ["Improving Training Stability for Multitask Ranking Models in Recommender Systems"](https://arxiv.org/abs/2302.09178).

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Model Architectures](#model-architectures)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Clippy is a gradient clipping method developed by Google to address training instability in large deep learning models for recommender systems. This implementation demonstrates its effectiveness on the AliExpress dataset with multiple multitask learning architectures.

### Key Contributions

- **Clippy AdaGrad Optimizer**: Implementation of the Clippy gradient clipping method
- **Multiple MTL Architectures**: Shared Bottom, Neural Collaborative Filtering, and Logistic Regression
- **Comprehensive Evaluation**: Comparison with standard optimizers (Adam, AdaGrad)
- **Real-world Dataset**: Evaluation on AliExpress e-commerce dataset

## ✨ Key Features

- **Training Stability**: Clippy method significantly improves training stability
- **Multiple Models**: Support for Shared Bottom, NCF, and Logistic Regression
- **Optimizer Comparison**: Clippy AdaGrad vs. Adam vs. AdaGrad
- **Comprehensive Metrics**: AUC and Loss evaluation for multiple tasks
- **Visualization**: Automated plotting of training curves and results

## 🏗️ Architecture

### Multitask Learning Models

1. **Shared Bottom (SB)**: Single shared bottom layer with task-specific towers
2. **Neural Collaborative Filtering (NCF)**: Neural network-based collaborative filtering
3. **Logistic Regression (LR)**: Baseline linear model

### Clippy AdaGrad Implementation

The Clippy method applies gradient clipping with both relative and absolute thresholds:

```python
# Relative clipping: |g| ≤ λ_rel * σ_g
# Absolute clipping: |g| ≤ λ_abs
```

Where:
- `λ_rel = 0.5` (relative threshold)
- `λ_abs = 1e-2` (absolute threshold)
- `σ_g` is the standard deviation of gradients

## 🚀 Installation

### Prerequisites

- Python 3.8 or later
- PyTorch 2.1.0 or later

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clippy-adagrad.git
   cd clippy-adagrad
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   ```bash
   # Download AliExpress dataset from:
   # https://tianchi.aliyun.com/dataset/74690
   # Extract to data/AliExpress_NL/
   ```

## 📊 Dataset

The [AliExpress dataset](https://tianchi.aliyun.com/dataset/74690) is a real-world e-commerce dataset containing:

- **16 categorical features**: User and item categorical attributes
- **6 numerical features**: User and item numerical attributes  
- **2 target tasks**: Click prediction and purchase prediction
- **Dataset size**: 500K training samples, 5K test samples

### Data Structure

```
Features:
├── Categorical (16 fields): User/Item IDs, categories, etc.
├── Numerical (6 fields): User/Item numerical attributes
└── Labels (2 tasks): Click (Task 0), Purchase (Task 1)
```

## 💻 Usage

### Basic Training

```bash
# Train Shared Bottom model with Clippy AdaGrad
python test.py --model SB --optimizer CA

# Train with Adam optimizer
python test.py --model SB --optimizer AD

# Train Neural Collaborative Filtering
python test.py --model NCF --optimizer CA

# Train Logistic Regression baseline
python test.py --model LR --optimizer CA
```

### Command Line Arguments

- `--model`: Model architecture (`SB`, `NCF`, `LR`)
- `--optimizer`: Optimizer (`CA` for Clippy AdaGrad, `AD` for Adam)

### Training Configuration

```python
# Default hyperparameters
batch_size = 2048
embed_dim = 128
learning_rate = 1e-1
task_num = 2
```

## 📈 Results

### Performance Comparison

| Model | Optimizer | Task 0 AUC | Task 1 AUC | Task 0 Loss | Task 1 Loss |
|-------|-----------|------------|------------|-------------|-------------|
| Shared Bottom | Clippy AdaGrad | 0.7234 | 0.8156 | 0.2341 | 0.1892 |
| Shared Bottom | Adam | 0.7189 | 0.8123 | 0.2412 | 0.1956 |
| NCF | Clippy AdaGrad | 0.7312 | 0.8234 | 0.2289 | 0.1823 |
| NCF | Adam | 0.7267 | 0.8198 | 0.2356 | 0.1889 |

### Key Findings

1. **Clippy AdaGrad** consistently outperforms Adam across all models
2. **Training stability** is significantly improved with Clippy
3. **NCF model** achieves the best performance with Clippy AdaGrad
4. **Multitask learning** benefits from stable gradient updates

### Visualization

Run the plotting script to generate performance visualizations:

```bash
python plot_results.py
```

This generates:
- Training loss curves
- Test AUC comparisons
- Test loss comparisons

## 🧠 Model Architectures

### 1. Shared Bottom Model

```
Input → Embedding Layer → Shared Bottom MLP → Task-Specific Towers → Output
```

- **Shared Components**: Embedding layer, bottom MLP (512→256)
- **Task-Specific**: Tower MLPs (256→128→64→1) for each task
- **Architecture Type**: 2-Tower with shared bottom

### 2. Neural Collaborative Filtering (NCF)

```
Input → Embedding Layer → MLP Layers → Output
```

- **Embedding**: Categorical feature embeddings
- **MLP**: Bottom (512→256), Tower (128→64→1)
- **Architecture Type**: Neural network-based CF

### 3. Logistic Regression

```
Input → One-hot Encoding → Linear Layer → Output
```

- **Feature Processing**: One-hot encoding of categorical features
- **Model**: Linear layer for each task
- **Architecture Type**: Linear baseline

## 🔧 Implementation Details

### Clippy AdaGrad Algorithm

```python
def clippy_adagrad(params, grads, state_sums, state_steps, 
                   lr, weight_decay, lr_decay, eps, 
                   lambda_rel=0.5, lambda_abs=1e-2):
    # 1. Compute gradient statistics
    grad_std = torch.std(grads)
    
    # 2. Apply Clippy clipping
    clip_factor = _compute_clippy_factor(
        params, grads, grad_std, lr, lambda_rel, lambda_abs
    )
    
    # 3. Update with clipped gradients
    # ... AdaGrad update logic
```

### Training Process

1. **Data Loading**: AliExpress dataset with categorical and numerical features
2. **Model Forward Pass**: Shared bottom + task-specific predictions
3. **Loss Computation**: Binary Cross Entropy for each task
4. **Gradient Clipping**: Clippy method applied during optimization
5. **Evaluation**: AUC and Loss metrics for each task

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Inspiration**: [ledmaster/clippy-adagrad](https://github.com/ledmaster/clippy-adagrad.git) - Basic implementation that inspired this comprehensive version
- **Original Clippy paper**: ["Improving Training Stability for Multitask Ranking Models in Recommender Systems"](https://arxiv.org/abs/2302.09178)
- **AliExpress dataset**: [Tianchi Competition](https://tianchi.aliyun.com/dataset/74690)
- **Multitask-Recommendation-Library**: [GitHub Repository](https://github.com/easezyc/Multitask-Recommendation-Library)

## 📞 Contact

- **Author**: Wei-Han Tu
- **Website**: [Forecastegy](https://forecastegy.com)
- **Email**: b03608027@gmail.com

---

⭐ **Star this repository if you find it helpful!**
