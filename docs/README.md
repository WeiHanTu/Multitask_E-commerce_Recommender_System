# Documentation

This directory contains comprehensive documentation for the Clippy AdaGrad implementation.

## 📚 Documentation Structure

- **[API Reference](api.md)**: Detailed API documentation for all classes and functions
- **[Architecture Guide](architecture.md)**: Deep dive into model architectures and Clippy implementation
- **[Tutorial](tutorial.md)**: Step-by-step tutorial for getting started
- **[Experiments](experiments.md)**: Detailed experiment results and analysis
- **[Contributing Guide](../CONTRIBUTING.md)**: Guidelines for contributors

## 🚀 Quick Start

1. **Installation**: See [main README](../README.md#installation)
2. **Basic Usage**: See [tutorial](tutorial.md)
3. **API Reference**: See [api.md](api.md)

## 📖 Key Concepts

### Clippy AdaGrad
Clippy is a gradient clipping method that applies both relative and absolute thresholds:

```python
# Relative clipping: |g| ≤ λ_rel * σ_g
# Absolute clipping: |g| ≤ λ_abs
```

### Multitask Learning
The project implements several MTL architectures:
- **Shared Bottom**: Single shared layer + task-specific towers
- **Neural Collaborative Filtering**: Neural network-based CF
- **Logistic Regression**: Linear baseline

### Dataset
AliExpress e-commerce dataset with:
- 16 categorical features
- 6 numerical features  
- 2 target tasks (click, purchase)

## 🔧 Development

For development guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md). 