import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CatsDogsDataset(Dataset):
    """Custom dataset for Cats vs Dogs classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DataPreprocessor:
    """Handle data preprocessing and dataset creation"""
    
    def __init__(self, data_dir, target_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.class_mapping = {'cat': 0, 'dog': 1}
        
    def load_dataset_from_folder(self, folder_path):
        """Load images from folder structure (cats/ and dogs/ subfolders)"""
        folder_path = Path(folder_path)
        image_paths = []
        labels = []
        
        for class_name, class_label in self.class_mapping.items():
            class_folder = folder_path / class_name
            if class_folder.exists():
                for img_file in class_folder.glob('*.jpg'):
                    image_paths.append(str(img_file))
                    labels.append(class_label)
                for img_file in class_folder.glob('*.png'):
                    image_paths.append(str(img_file))
                    labels.append(class_label)
        
        logger.info(f"Loaded {len(image_paths)} images from {folder_path}")
        return image_paths, labels
    
    def preprocess_images(self, image_paths, output_dir):
        """Preprocess images to target size and save to output directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_paths = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
                filename = Path(img_path).name
                output_path = output_dir / filename
                img_resized.save(output_path)
                processed_paths.append(str(output_path))
                
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
        
        logger.info(f"Preprocessed {len(processed_paths)} images")
        return processed_paths
    
    def create_data_splits(self, image_paths, labels, test_size=0.1, val_size=0.1, random_state=42):
        """Create train/validation/test splits"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def get_data_transforms(self):
        """Get data augmentation and normalization transforms"""
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure consistent size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure consistent size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return {
            'train': train_transform,
            'val': val_test_transform,
            'test': val_test_transform
        }
    
    def create_dataloaders(self, data_splits, batch_size=32, num_workers=4):
        """Create PyTorch DataLoaders"""
        transforms = self.get_data_transforms()
        dataloaders = {}
        
        for split_name, (paths, labels) in data_splits.items():
            dataset = CatsDogsDataset(
                image_paths=paths,
                labels=labels,
                transform=transforms[split_name]
            )
            
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                num_workers=num_workers,
                pin_memory=True
            )
        
        return dataloaders

def validate_preprocessing():
    """Validate preprocessing functions"""
    preprocessor = DataPreprocessor("data/raw")
    
    # Test class mapping
    assert preprocessor.class_mapping == {'cat': 0, 'dog': 1}
    
    # Test target size
    assert preprocessor.target_size == (224, 224)
    
    # Test transforms
    transforms = preprocessor.get_data_transforms()
    assert 'train' in transforms
    assert 'val' in transforms
    assert 'test' in transforms
    
    logger.info("Preprocessing validation passed")

if __name__ == "__main__":
    validate_preprocessing()
