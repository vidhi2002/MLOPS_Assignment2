import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import io
import json

from mlops_project.models.cnn_model import ModelUtils
from mlops_project.data.preprocessing import DataPreprocessor

class TestInferenceFunctions:
    """Test cases for inference functions"""
    
    @pytest.fixture
    def sample_model(self):
        """Create and save a sample model for testing"""
        model = ModelUtils.get_model(num_classes=2)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Save model
        ModelUtils.save_model(model, temp_path)
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a simple RGB image
        import numpy as np
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes
    
    def test_model_loading(self, sample_model):
        """Test model loading functionality"""
        # Test loading the model
        loaded_model = ModelUtils.load_model(sample_model, num_classes=2)
        
        assert isinstance(loaded_model, torch.nn.Module)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        assert output.shape == (1, 2)
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing for inference"""
        # Load image from bytes
        img = Image.open(sample_image).convert('RGB')
        
        # Test preprocessing function (similar to what's in app.py)
        def preprocess_image(image):
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor = transform(image).unsqueeze(0)
            return tensor
        
        processed = preprocess_image(img)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 224, 224)
    
    def test_prediction_pipeline(self, sample_model, sample_image):
        """Test complete prediction pipeline"""
        # Load model
        model = ModelUtils.load_model(sample_model, num_classes=2)
        model.eval()
        
        # Preprocess image
        img = Image.open(sample_image).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Validate results
        assert isinstance(predicted_class.item(), int)
        assert predicted_class.item() in [0, 1]
        assert 0 <= confidence.item() <= 1
        assert probabilities.shape == (1, 2)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    
    def test_prediction_response_format(self, sample_model, sample_image):
        """Test prediction response format"""
        # Load model
        model = ModelUtils.load_model(sample_model, num_classes=2)
        model.eval()
        
        # Preprocess and predict
        img = Image.open(sample_image).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Create response in expected format
        class_mapping = {0: 'cat', 1: 'dog'}
        probabilities_np = probabilities.cpu().numpy()[0]
        
        response = {
            'predicted_class': class_mapping[predicted_class.item()],
            'confidence': float(confidence.item()),
            'probabilities': {
                'cat': float(probabilities_np[0]),
                'dog': float(probabilities_np[1])
            }
        }
        
        # Validate response format
        assert 'predicted_class' in response
        assert 'confidence' in response
        assert 'probabilities' in response
        
        assert response['predicted_class'] in ['cat', 'dog']
        assert isinstance(response['confidence'], float)
        assert 0 <= response['confidence'] <= 1
        assert isinstance(response['probabilities'], dict)
        assert 'cat' in response['probabilities']
        assert 'dog' in response['probabilities']
        assert len(response['probabilities']) == 2
        
        # Check that probabilities sum to 1
        prob_sum = response['probabilities']['cat'] + response['probabilities']['dog']
        assert abs(prob_sum - 1.0) < 1e-6
    
    def test_error_handling_invalid_image(self):
        """Test error handling for invalid images"""
        # Test with non-image data
        invalid_data = b"This is not an image"
        
        with pytest.raises(Exception):
            img = Image.open(io.BytesIO(invalid_data))
            img.convert('RGB')
    
    def test_error_handling_invalid_model_path(self):
        """Test error handling for invalid model path"""
        invalid_path = "/nonexistent/path/model.pth"
        
        with pytest.raises(FileNotFoundError):
            ModelUtils.load_model(invalid_path, num_classes=2)
    
    def test_batch_predictions(self, sample_model):
        """Test batch prediction functionality"""
        model = ModelUtils.load_model(sample_model, num_classes=2)
        model.eval()
        
        # Create batch of dummy inputs
        batch_size = 4
        dummy_batch = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(dummy_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Validate batch results
        assert outputs.shape == (batch_size, 2)
        assert probabilities.shape == (batch_size, 2)
        assert predicted_classes.shape == (batch_size,)
        
        # Check that all predictions are valid
        for pred in predicted_classes:
            assert pred.item() in [0, 1]
        
        # Check that probabilities sum to 1 for each sample
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size))

if __name__ == "__main__":
    pytest.main([__file__])
