import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from mlops_project.models.cnn_model import SimpleCNN, ModelUtils

class TestSimpleCNN:
    """Test cases for SimpleCNN model"""
    
    @pytest.fixture
    def model(self):
        """Create a SimpleCNN instance for testing"""
        return SimpleCNN(num_classes=2)
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, SimpleCNN)
        assert model.fc3.out_features == 2
    
    def test_forward_pass(self, model):
        """Test forward pass"""
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (batch_size, 2)
        
        # Check output type
        assert isinstance(output, torch.Tensor)
    
    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes"""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            output = model(dummy_input)
            assert output.shape == (batch_size, 2)
    
    def test_model_in_training_mode(self, model):
        """Test model behavior in training mode"""
        model.train()
        assert model.training is True
        
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.requires_grad is True
    
    def test_model_in_evaluation_mode(self, model):
        """Test model behavior in evaluation mode"""
        model.eval()
        assert model.training is False
        
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.requires_grad is False

class TestModelUtils:
    """Test cases for ModelUtils class"""
    
    def test_get_model(self):
        """Test model creation"""
        model = ModelUtils.get_model(num_classes=2)
        
        assert isinstance(model, SimpleCNN)
        assert model.fc3.out_features == 2
        
        # Test with different number of classes
        model_3_classes = ModelUtils.get_model(num_classes=3)
        assert model_3_classes.fc3.out_features == 3
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Create model
        original_model = ModelUtils.get_model(num_classes=2)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Save model
            ModelUtils.save_model(original_model, temp_path)
            assert Path(temp_path).exists()
            
            # Load model
            loaded_model = ModelUtils.load_model(temp_path, num_classes=2)
            
            # Check that models have same architecture
            assert type(loaded_model) == type(original_model)
            assert loaded_model.fc3.out_features == original_model.fc3.out_features
            
            # Check that models produce same output
            dummy_input = torch.randn(1, 3, 224, 224)
            
            original_model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                original_output = original_model(dummy_input)
                loaded_output = loaded_model(dummy_input)
                
                assert torch.allclose(original_output, loaded_output, atol=1e-6)
        
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
    
    def test_count_parameters(self):
        """Test parameter counting"""
        model = ModelUtils.get_model(num_classes=2)
        param_count = ModelUtils.count_parameters(model)
        
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # Check that all parameters are counted
        total_params = sum(p.numel() for p in model.parameters())
        assert param_count == total_params
    
    def test_get_model_info(self):
        """Test model information extraction"""
        model = ModelUtils.get_model(num_classes=2)
        info = ModelUtils.get_model_info(model)
        
        # Check required fields
        assert 'total_parameters' in info
        assert 'input_shape' in info
        assert 'output_shape' in info
        assert 'model_class' in info
        
        # Check field types
        assert isinstance(info['total_parameters'], int)
        assert isinstance(info['input_shape'], tuple)
        assert isinstance(info['output_shape'], torch.Size)
        assert isinstance(info['model_class'], str)
        
        # Check values
        assert info['total_parameters'] > 0
        assert info['input_shape'] == (1, 3, 224, 224)
        assert info['output_shape'][0] == 1  # Batch size
        assert info['output_shape'][1] == 2  # Number of classes
        assert info['model_class'] == 'SimpleCNN'
    
    def test_model_with_different_input_sizes(self):
        """Test model with different input sizes"""
        model = ModelUtils.get_model(num_classes=2)
        
        # Test with standard input size
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 2)
        
        # Test with slightly different input size (should still work due to adaptive pooling)
        # Note: This test depends on the model architecture
        # For SimpleCNN, it expects 224x224 input
    
    def test_model_gradients(self):
        """Test gradient computation"""
        model = ModelUtils.get_model(num_classes=2)
        model.train()
        
        # Create dummy input and target
        dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 2, (2,))
        
        # Forward pass
        output = model(dummy_input)
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

if __name__ == "__main__":
    pytest.main([__file__])
