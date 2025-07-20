# Contributing to Clippy AdaGrad

Thank you for your interest in contributing to the Clippy AdaGrad implementation! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Types of Contributions

We welcome contributions in the following areas:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new model architectures or optimizers
3. **Documentation**: Improve documentation and examples
4. **Performance**: Optimize existing implementations
5. **Testing**: Add unit tests and integration tests
6. **Research**: Implement new multitask learning methods

### Before You Start

1. **Check existing issues**: Look for existing issues or pull requests
2. **Discuss changes**: Open an issue to discuss major changes
3. **Fork the repository**: Create your own fork for development

## Development Setup

### Prerequisites

- Python 3.8 or later
- PyTorch 2.1.0 or later
- Git

### Local Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clippy-adagrad.git
   cd clippy-adagrad
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**:
   ```bash
   pip install pytest black flake8 mypy
   ```

5. **Verify installation**:
   ```bash
   python test.py --model SB --optimizer CA
   ```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black formatter)
- **Import organization**: Standard library, third-party, local
- **Docstrings**: Google style docstrings
- **Type hints**: Use type hints for all function parameters and return values

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Run flake8
flake8 .

# Ignore specific errors if needed
flake8 --ignore=E501,W503 .
```

### Type Checking

We use [mypy](http://mypy-lang.org/) for static type checking:

```bash
# Run mypy
mypy .

# Run with strict mode
mypy --strict .
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=.

# Run with verbose output
pytest -v
```

### Writing Tests

- **Test files**: Place in `tests/` directory
- **Test functions**: Prefix with `test_`
- **Test classes**: Prefix with `Test`
- **Fixtures**: Use pytest fixtures for common setup

Example test structure:

```python
import pytest
import torch
from models.sharedbottom import SharedBottomModel

class TestSharedBottomModel:
    def test_initialization(self):
        field_dims = [10, 20, 30]
        numerical_num = 5
        model = SharedBottomModel(field_dims, numerical_num)
        assert model is not None
    
    def test_forward_pass(self):
        field_dims = [10, 20, 30]
        numerical_num = 5
        model = SharedBottomModel(field_dims, numerical_num)
        
        batch_size = 32
        categorical_fields = torch.randint(0, 10, (batch_size, len(field_dims)))
        numerical_fields = torch.randn(batch_size, numerical_num)
        
        outputs = model(categorical_fields, numerical_fields)
        assert len(outputs) == 2  # Two tasks
        assert outputs[0].shape == (batch_size, 1)
        assert outputs[1].shape == (batch_size, 1)
```

### Test Coverage

We aim for at least 80% test coverage. To check coverage:

```bash
pytest --cov=. --cov-report=html
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: Run all tests locally
2. **Format code**: Run Black formatter
3. **Check linting**: Run flake8 and mypy
4. **Update documentation**: Update relevant documentation
5. **Add tests**: Add tests for new functionality

### Pull Request Guidelines

1. **Title**: Use clear, descriptive titles
2. **Description**: Explain what and why, not how
3. **Related issues**: Link to related issues
4. **Screenshots**: Include for UI changes
5. **Breaking changes**: Clearly mark breaking changes

Example PR description:

```markdown
## Description
Implements new MMoE (Mixture of Multi-Expert) model architecture for multitask learning.

## Changes
- Added `MMoEModel` class in `models/mmoe.py`
- Added corresponding tests in `tests/test_mmoe.py`
- Updated documentation in `docs/api.md`

## Testing
- [x] All tests pass
- [x] Added unit tests for MMoE model
- [x] Verified performance on AliExpress dataset

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: At least one maintainer reviews
3. **Address feedback**: Respond to review comments
4. **Merge**: Maintainer merges after approval

## Reporting Bugs

### Before Reporting

1. **Check existing issues**: Search for similar issues
2. **Reproduce the bug**: Ensure it's reproducible
3. **Gather information**: Collect relevant details

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.8.10]
- PyTorch: [e.g., 2.1.0]
- Other dependencies: [list relevant packages]

## Additional Information
Any additional context, logs, or screenshots.
```

## Feature Requests

### Before Requesting

1. **Check existing features**: Ensure it's not already implemented
2. **Research**: Look for similar features in other projects
3. **Consider impact**: Think about maintenance burden

### Feature Request Template

```markdown
## Feature Description
Brief description of the requested feature.

## Use Case
Why this feature is needed and how it would be used.

## Proposed Implementation
High-level description of how it could be implemented.

## Alternatives Considered
Other approaches that were considered.

## Additional Information
Any additional context or examples.
```

## Code Review Guidelines

### For Reviewers

- **Be constructive**: Provide helpful, specific feedback
- **Focus on code**: Avoid personal comments
- **Explain reasoning**: Explain why changes are suggested
- **Be timely**: Respond within a reasonable timeframe

### For Authors

- **Be responsive**: Address review comments promptly
- **Be open**: Consider feedback constructively
- **Ask questions**: Clarify unclear feedback
- **Be patient**: Reviews take time

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Tagged release

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the docs directory
- **Examples**: Look at existing code examples

## Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **Release notes**: Credit for significant contributions
- **Documentation**: Attribution for documentation contributions

Thank you for contributing to Clippy AdaGrad! 🚀 