"""
Simple test to verify basic functionality.
"""

import pytest
import torch
import torch.nn as nn


def test_torch_import():
    """Test that PyTorch can be imported."""
    assert torch is not None
    print(f"PyTorch version: {torch.__version__}")


def test_simple_model():
    """Test that a simple model can be created."""
    model = nn.Linear(10, 1)
    assert model is not None
    
    # Test forward pass
    x = torch.randn(5, 10)
    output = model(x)
    assert output.shape == (5, 1)


def test_simple_optimizer():
    """Test that a simple optimizer can be created."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    assert optimizer is not None


def test_basic_training_step():
    """Test a basic training step."""
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    # Create dummy data
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # If we reach here, everything works
    assert True


if __name__ == "__main__":
    pytest.main([__file__]) 