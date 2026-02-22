import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Simple CNN model for Cats vs Dogs classification"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling operations: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def get_model(num_classes=2, pretrained=False):
        """Get model instance"""
        model = SimpleCNN(num_classes=num_classes)
        
        if pretrained:
            # In a real scenario, you might load pretrained weights
            logger.info("Using pretrained weights (placeholder)")
        
        return model
    
    @staticmethod
    def save_model(model, filepath):
        """Save model state dict"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, num_classes=2):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    @staticmethod
    def count_parameters(model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_info(model):
        """Get model information"""
        total_params = ModelUtils.count_parameters(model)
        
        # Create a dummy input to get output shape
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        return {
            'total_parameters': total_params,
            'input_shape': (1, 3, 224, 224),
            'output_shape': output.shape,
            'model_class': model.__class__.__name__
        }

def test_model():
    """Test model functionality"""
    model = ModelUtils.get_model(num_classes=2)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
    
    # Test model info
    info = ModelUtils.get_model_info(model)
    assert 'total_parameters' in info
    assert info['total_parameters'] > 0
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        ModelUtils.save_model(model, f.name)
        loaded_model = ModelUtils.load_model(f.name, num_classes=2)
        
        # Test that loaded model gives same output
        with torch.no_grad():
            original_output = model(dummy_input)
            loaded_output = loaded_model(dummy_input)
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
    
    logger.info("Model tests passed")

if __name__ == "__main__":
    test_model()
