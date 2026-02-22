import pytest
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path

from mlops_project.data.preprocessing import DataPreprocessor, CatsDogsDataset

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance for testing"""
        return DataPreprocessor("data/raw")
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create sample images
        cat_dir = Path(temp_dir) / "cat"
        dog_dir = Path(temp_dir) / "dog"
        cat_dir.mkdir(exist_ok=True)
        dog_dir.mkdir(exist_ok=True)
        
        # Create sample cat images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(cat_dir / f"cat_{i}.jpg")
        
        # Create sample dog images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(dog_dir / f"dog_{i}.jpg")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, preprocessor):
        """Test DataPreprocessor initialization"""
        assert preprocessor.data_dir.name == "raw"
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.class_mapping == {'cat': 0, 'dog': 1}
    
    def test_load_dataset_from_folder(self, preprocessor, sample_images):
        """Test loading dataset from folder structure"""
        image_paths, labels = preprocessor.load_dataset_from_folder(sample_images)
        
        assert len(image_paths) == 6  # 3 cats + 3 dogs
        assert len(labels) == 6
        assert sum(labels) == 3  # 3 dogs (label 1) + 3 cats (label 0)
        assert all(path.endswith('.jpg') for path in image_paths)
    
    def test_preprocess_images(self, preprocessor, sample_images):
        """Test image preprocessing"""
        image_paths, _ = preprocessor.load_dataset_from_folder(sample_images)
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        
        try:
            processed_paths = preprocessor.preprocess_images(image_paths, output_dir)
            
            assert len(processed_paths) == len(image_paths)
            
            # Check that processed images have correct size
            for path in processed_paths:
                img = Image.open(path)
                assert img.size == preprocessor.target_size
                assert img.mode == 'RGB'
        
        finally:
            # Cleanup with retry for Windows file locking
            import time
            for _ in range(3):
                try:
                    shutil.rmtree(output_dir)
                    break
                except PermissionError:
                    time.sleep(0.1)
    
    def test_create_data_splits(self, preprocessor):
        """Test data split creation"""
        # Create dummy data
        image_paths = [f"image_{i}.jpg" for i in range(100)]
        labels = [i % 2 for i in range(100)]  # Alternate between 0 and 1
        
        splits = preprocessor.create_data_splits(image_paths, labels)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check split sizes (approximately 80/10/10)
        train_size = len(splits['train'][0])
        val_size = len(splits['val'][0])
        test_size = len(splits['test'][0])
        
        assert train_size == 80
        assert val_size == 10
        assert test_size == 10
        assert train_size + val_size + test_size == 100
    
    def test_get_data_transforms(self, preprocessor):
        """Test data transforms"""
        transforms = preprocessor.get_data_transforms()
        
        assert 'train' in transforms
        assert 'val' in transforms
        assert 'test' in transforms
        
        # Test transform on sample image
        sample_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test train transform
        transformed = transforms['train'](sample_img)
        assert transformed.shape == (3, 224, 224)
        
        # Test val transform
        transformed = transforms['val'](sample_img)
        assert transformed.shape == (3, 224, 224)
        
        # Test test transform
        transformed = transforms['test'](sample_img)
        assert transformed.shape == (3, 224, 224)

class TestCatsDogsDataset:
    """Test cases for CatsDogsDataset class"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing"""
        image_paths = [f"image_{i}.jpg" for i in range(10)]
        labels = [i % 2 for i in range(10)]
        
        # Create actual image files
        temp_dir = tempfile.mkdtemp()
        actual_paths = []
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            img_path = Path(temp_dir) / f"image_{i}.jpg"
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(img_path)
            actual_paths.append(str(img_path))
        
        dataset = CatsDogsDataset(actual_paths, labels)
        
        yield dataset, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_length(self, sample_dataset):
        """Test dataset length"""
        dataset, _ = sample_dataset
        assert len(dataset) == 10
    
    def test_dataset_getitem(self, sample_dataset):
        """Test dataset item retrieval"""
        dataset, _ = sample_dataset
        
        image, label = dataset[0]
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
        assert isinstance(label, int)
        assert label in [0, 1]
    
    def test_dataset_with_transforms(self, sample_dataset):
        """Test dataset with transforms"""
        import torch
        import torchvision.transforms as transforms
        
        dataset, _ = sample_dataset
        
        # Apply transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset.transform = transform
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)  # Should be tensor
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

if __name__ == "__main__":
    pytest.main([__file__])
