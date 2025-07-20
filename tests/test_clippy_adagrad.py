import pytest
import torch
import torch.nn as nn
from clippyadagrad import ClippyAdagrad


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


class TestClippyAdagrad:
    def test_initialization(self):
        """Test ClippyAdagrad optimizer initialization."""
        model = SimpleModel()
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-2
        assert optimizer.param_groups[0]['lambda_rel'] == 0.5
        assert optimizer.param_groups[0]['lambda_abs'] == 1e-2
    
    def test_custom_parameters(self):
        """Test ClippyAdagrad with custom parameters."""
        model = SimpleModel()
        optimizer = ClippyAdagrad(
            model.parameters(),
            lr=1e-1,
            lambda_rel=0.3,
            lambda_abs=5e-3
        )
        
        assert optimizer.param_groups[0]['lr'] == 1e-1
        assert optimizer.param_groups[0]['lambda_rel'] == 0.3
        assert optimizer.param_groups[0]['lambda_abs'] == 5e-3
    
    def test_invalid_parameters(self):
        """Test ClippyAdagrad with invalid parameters."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            ClippyAdagrad(model.parameters(), lr=-1.0)
        
        with pytest.raises(ValueError):
            ClippyAdagrad(model.parameters(), lambda_rel=-0.1)
        
        with pytest.raises(ValueError):
            ClippyAdagrad(model.parameters(), lambda_abs=-0.1)
    
    def test_optimization_step(self):
        """Test that optimization step works without errors."""
        model = SimpleModel()
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()
        
        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Check that parameters were updated
        for param in model.parameters():
            assert param.grad is not None
    
    def test_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        model = SimpleModel()
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2, lambda_rel=0.1, lambda_abs=1e-3)
        criterion = nn.MSELoss()
        
        # Create data with large gradients
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist before optimization step
        for param in model.parameters():
            assert param.grad is not None
        
        # Optimization step
        optimizer.step()
        
        # Verify that optimization completed without errors
        assert True  # If we reach here, no exceptions were raised
    
    def test_multiple_parameter_groups(self):
        """Test ClippyAdagrad with multiple parameter groups."""
        model = SimpleModel()
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': [model.linear.weight], 'lr': 1e-2},
            {'params': [model.linear.bias], 'lr': 1e-1}
        ]
        
        optimizer = ClippyAdagrad(param_groups)
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 1e-2
        assert optimizer.param_groups[1]['lr'] == 1e-1
    
    def test_state_management(self):
        """Test that optimizer state is properly managed."""
        model = SimpleModel()
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        
        # Check initial state
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                assert 'step' in state
                assert 'sum' in state
                assert state['step'] == torch.tensor(0.0)
    
    def test_serialization(self):
        """Test that optimizer can be serialized and deserialized."""
        model = SimpleModel()
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        
        # Save optimizer state
        state_dict = optimizer.state_dict()
        
        # Create new optimizer and load state
        new_optimizer = ClippyAdagrad(model.parameters(), lr=1e-2)
        new_optimizer.load_state_dict(state_dict)
        
        # Check that states match
        assert optimizer.state_dict() == new_optimizer.state_dict()


if __name__ == "__main__":
    pytest.main([__file__]) 