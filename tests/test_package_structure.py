"""
Test package structure and basic functionality.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os


def test_package_import():
    """Test that the package can be imported correctly."""
    try:
        # Test direct import
        from clippyadagrad import ClippyAdagrad, clippy_adagrad
        assert ClippyAdagrad is not None
        assert clippy_adagrad is not None
    except ImportError as e:
        pytest.fail(f"Failed to import package: {e}")


def test_optimizer_creation():
    """Test that ClippyAdagrad optimizer can be created."""
    try:
        from clippyadagrad import ClippyAdagrad
        
        # Create a simple model
        model = nn.Linear(10, 1)
        
        # Create optimizer with default parameters
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        
        # Check that optimizer was created successfully
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-2
        
    except Exception as e:
        pytest.fail(f"Failed to create optimizer: {e}")


def test_optimizer_step():
    """Test that optimizer step works correctly."""
    try:
        from clippyadagrad import ClippyAdagrad
        
        # Create a simple model
        model = nn.Linear(10, 1)
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
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
        
        # Optimization step
        optimizer.step()
        
        # If we reach here, the optimizer works
        assert True
        
    except Exception as e:
        pytest.fail(f"Optimizer step failed: {e}")


def test_custom_parameters():
    """Test ClippyAdagrad with custom parameters."""
    try:
        from clippyadagrad import ClippyAdagrad
        
        model = nn.Linear(10, 1)
        
        # Test with custom Clippy parameters
        optimizer = ClippyAdagrad(
            model.parameters(),
            lr=1e-1,
            lambda_rel=0.3,
            lambda_abs=5e-3
        )
        
        # Check parameters
        assert optimizer.param_groups[0]['lr'] == 1e-1
        assert optimizer.param_groups[0]['lambda_rel'] == 0.3
        assert optimizer.param_groups[0]['lambda_abs'] == 5e-3
        
    except Exception as e:
        pytest.fail(f"Custom parameters test failed: {e}")


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    try:
        from clippyadagrad import ClippyAdagrad
        
        model = nn.Linear(10, 1)
        
        # Test negative learning rate
        with pytest.raises(ValueError):
            ClippyAdagrad(model.parameters(), lr=-1.0)
        
        # Test negative lambda_rel
        with pytest.raises(ValueError):
            ClippyAdagrad(model.parameters(), lambda_rel=-0.1)
        
        # Test negative lambda_abs
        with pytest.raises(ValueError):
            ClippyAdagrad(model.parameters(), lambda_abs=-0.1)
            
    except Exception as e:
        pytest.fail(f"Invalid parameters test failed: {e}")


def test_package_version():
    """Test that package version is accessible."""
    try:
        import clippyadagrad
        assert hasattr(clippyadagrad, '__version__')
        assert clippyadagrad.__version__ == "1.0.0"
    except Exception as e:
        pytest.fail(f"Package version test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 