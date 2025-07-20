"""
Test that all imports work correctly.
"""

import pytest
import torch
import torch.nn as nn


def test_clippy_adagrad_import():
    """Test that ClippyAdagrad can be imported."""
    try:
        from clippyadagrad import ClippyAdagrad
        assert ClippyAdagrad is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ClippyAdagrad: {e}")


def test_clippy_adagrad_functional_import():
    """Test that clippy_adagrad function can be imported."""
    try:
        from clippyadagrad import clippy_adagrad
        assert clippy_adagrad is not None
    except ImportError as e:
        pytest.fail(f"Failed to import clippy_adagrad function: {e}")


def test_basic_optimizer_usage():
    """Test basic usage of ClippyAdagrad optimizer."""
    try:
        from clippyadagrad import ClippyAdagrad
        
        # Create a simple model
        model = nn.Linear(10, 1)
        
        # Create optimizer
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        
        # Create dummy data
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # If we reach here, everything works
        assert True
        
    except Exception as e:
        pytest.fail(f"Basic optimizer usage failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 