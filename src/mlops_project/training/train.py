import os
import sys
import argparse
import logging
from pathlib import Path
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch

from src.data.preprocessing import DataPreprocessor
from src.models.cnn_model import ModelUtils, SimpleCNN

class ImprovedCNN(nn.Module):
    """Improved CNN model for Cats vs Dogs classification with better accuracy"""

    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()

        # Enhanced architecture with more layers and better feature extraction
        self.features = nn.Sequential(
            # Block 1: 64 filters
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3: 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Enhanced classifier with more layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with better initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """Model training class with MLflow integration"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize MLflow
        mlflow.set_experiment(config['experiment_name'])
        
    def setup_model(self):
        """Setup model, loss function, and optimizer"""
        self.model = ImprovedCNN(num_classes=self.config['num_classes'])
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        logger.info(f"Model setup complete. Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(self.config)
            
            for epoch in range(self.config['num_epochs']):
                logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
                
                # Train
                train_loss, train_acc = self.train_epoch(train_loader)
                
                # Validate
                val_loss, val_acc, val_preds, val_targets = self.validate_epoch(val_loader)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                # Store for plotting
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save improved model
                    model_path = Path(self.config['model_save_dir']) / 'best_model.pth'
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'model_class': 'ImprovedCNN',
                        'num_classes': self.config['num_classes'],
                        'input_size': (3, 224, 224),
                    }, str(model_path))
                    mlflow.pytorch.log_model(self.model, "model")
                    
                    # Log validation metrics
                    val_report = classification_report(val_targets, val_preds, output_dict=True)
                    mlflow.log_metrics({
                        'val_precision': val_report['weighted avg']['precision'],
                        'val_recall': val_report['weighted avg']['recall'],
                        'val_f1': val_report['weighted avg']['f1-score']
                    })
            
            # Plot and save training curves
            self.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(val_targets, val_preds)
            
            logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
            
            return best_val_acc
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path(self.config['model_save_dir']) / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'training_curves.png')
        mlflow.log_artifact(str(plots_dir / 'training_curves.png'))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plots_dir = Path(self.config['model_save_dir']) / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'confusion_matrix.png')
        mlflow.log_artifact(str(plots_dir / 'confusion_matrix.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'experiment_name': 'cats-dogs-classification',
        'num_classes': 2,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'model_save_dir': 'models',
        'data_dir': args.data_dir,
        'test_size': 0.1,
        'val_size': 0.1,
        'random_state': 42,
        'num_workers': 4
    }
    
    # Override with config file if exists
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Setup data
    preprocessor = DataPreprocessor(config['data_dir'])
    
    # Load dataset
    logger.info("Loading dataset...")
    image_paths, labels = preprocessor.load_dataset_from_folder(config['data_dir'])
    
    # Split dataset
    data_splits = preprocessor.create_data_splits(
        image_paths, labels, 
        test_size=config['test_size'], 
        val_size=config['val_size'],
        random_state=config['random_state']
    )
    
    # Create data loaders
    dataloaders = preprocessor.create_dataloaders(
        data_splits, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers']
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Initialize trainer
    trainer = Trainer(config)
    trainer.setup_model()
    
    # Load actual data and start training
    trainer.train(train_loader, val_loader)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
