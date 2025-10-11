#!/usr/bin/env python3
"""
Main training script with robust error handling for model downloads.
Uses ResNet50 as first model and MobileNetV3 as second model.
"""

import os
import random
import json
import sys
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
os.environ['TIMM_DOWNLOAD_TIMEOUT'] = '30'
os.environ['TIMM_DOWNLOAD_RETRY'] = '2'

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from config import DataConfig, ModelConfig, TrainConfig, ExperimentConfig, EXPERIMENT_CONFIGS


class ModelTrainer:
    """Model trainer with robust error handling."""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.data_cfg = experiment_config.data
        self.model_cfg = experiment_config.model
        self.train_cfg = experiment_config.train
        self.set_seed()
        
    def set_seed(self):
        """Fix all random generators for reproducibility."""
        random.seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)
        torch.manual_seed(self.train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train_cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def seed_worker(self, worker_id):
        """Seed worker for DataLoader reproducibility."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def setup_data_directories(self):
        """Create necessary directories."""
        data_dirs = [
            "data/raw/minivan",
            "data/raw/sedan", 
            "data/raw/wagon",
            "artifacts/models",
            "artifacts/plots"
        ]
        
        for dir_path in data_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    def check_data_exists(self, data_root):
        if not Path(data_root).exists():
            print(f"Directory {data_root} does not exist!")
            return False
            
        subdirs = [d for d in Path(data_root).iterdir() if d.is_dir()]
        if not subdirs:
            print(f"No class directories in {data_root}!")
            return False
            
        print(f"Data found in: {data_root}")
        total_images = 0
        
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
            images = [f for f in subdir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
            
            print(f" {subdir.name}: {len(images)} images")
            total_images += len(images)
            
        print(f"Total images: {total_images}")
        
        if total_images < 30:
            print("Less than 30 images total - may affect model performance")
            
        return total_images > 0
        
    def get_transforms(self):
        if self.data_cfg.augmentation:
            train_t = transforms.Compose([
                transforms.RandomResizedCrop(self.data_cfg.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_t = transforms.Compose([
                transforms.Resize((self.data_cfg.img_size, self.data_cfg.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        val_t = transforms.Compose([
            transforms.Resize((self.data_cfg.img_size, self.data_cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_t, val_t

    def create_data_loaders(self):
        """Create DataLoaders with full reproducibility."""
        train_t, val_t = self.get_transforms()
        
        full_dataset = ImageFolder(self.data_cfg.data_root)
        indices = list(range(len(full_dataset)))
        labels = [s[1] for s in full_dataset.samples]
        
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=self.data_cfg.test_ratio, 
            stratify=labels, 
            random_state=self.train_cfg.seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, 
            test_size=self.data_cfg.val_ratio/(1-self.data_cfg.test_ratio),
            stratify=[labels[i] for i in train_idx], 
            random_state=self.train_cfg.seed
        )
        
        train_ds = ImageFolder(self.data_cfg.data_root, transform=train_t)
        val_ds = ImageFolder(self.data_cfg.data_root, transform=val_t)
        test_ds = ImageFolder(self.data_cfg.data_root, transform=val_t)
        
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)
        test_subset = Subset(test_ds, test_idx)
        
        generator = torch.Generator()
        generator.manual_seed(self.train_cfg.seed)
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=self.data_cfg.batch_size, 
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            worker_init_fn=self.seed_worker,
            generator=generator,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=self.data_cfg.batch_size, 
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_subset, 
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True
        )
        
        self.classes = full_dataset.classes
        print(f"Classes: {self.classes}")
        print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        return train_loader, val_loader, test_loader

    def create_model(self):
        """Create model with robust error handling for downloads."""
        print(f"Creating model: {self.model_cfg.model_name}")
        print(f"Pretrained: {self.model_cfg.pretrained}")
        
        try:
            model = timm.create_model(
                self.model_cfg.model_name, 
                pretrained=self.model_cfg.pretrained, 
                num_classes=self.model_cfg.num_classes
            )
            print(f"Successfully created {self.model_cfg.model_name}")
            
        except Exception as e:
            print(f"Failed to download {self.model_cfg.model_name}: {e}")
            print("Falling back to randomly initialized model...")
            
            model = timm.create_model(
                self.model_cfg.model_name, 
                pretrained=False,  
                num_classes=self.model_cfg.num_classes
            )
            print(f"Created {self.model_cfg.model_name} with random initialization")
        
        if self.model_cfg.freeze_backbone:
            frozen_count = 0
            trainable_count = 0
            
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in ['head', 'fc', 'classifier']):
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
                    
            print(f"Freezing: {frozen_count} frozen, {trainable_count} trainable")
            print(f"Unfreeze at epoch: {self.model_cfg.unfreeze_epoch}")
                    
        return model

    def unfreeze_model(self, model):
        """Unfreeze all model parameters."""
        print("Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True

    def train_epoch(self, model, loader, criterion, optimizer, device):
        """Single training epoch."""
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
        avg_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        
        return avg_loss, acc

    def validate(self, model, loader, criterion, device):
        """Validate model."""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                
        avg_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, 
                                     target_names=self.classes, 
                                     zero_division=0)
        
        return avg_loss, acc, cm, report

    def train(self):
        print(f"\nTraining {self.model_cfg.model_name}")
        
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        model = self.create_model()
        model = model.to(self.train_cfg.device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.model_cfg.learning_rate,
            weight_decay=self.model_cfg.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.train_cfg.epochs)
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
        best_val_acc = 0.0
        
        print(f"\nTraining for {self.train_cfg.epochs} epochs on {self.train_cfg.device}")
        
        for epoch in range(self.train_cfg.epochs):
            if epoch == self.model_cfg.unfreeze_epoch and self.model_cfg.freeze_backbone:
                print(f"\nEpoch {epoch}: Unfreezing backbone")
                self.unfreeze_model(model)
                optimizer = AdamW(
                    model.parameters(),
                    lr=self.model_cfg.learning_rate/10,
                    weight_decay=self.model_cfg.weight_decay
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=self.train_cfg.epochs - epoch)
            
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, self.train_cfg.device
            )
            
            val_loss, val_acc, cm, report = self.validate(
                model, val_loader, criterion, self.train_cfg.device
            )
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if (epoch + 1) % self.train_cfg.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.train_cfg.epochs}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(model, epoch, val_acc)
                if (epoch + 1) % self.train_cfg.log_interval == 0:
                    print(f"New best model Val Acc: {val_acc:.4f}")
        
        final_metrics = self.final_evaluation(model, test_loader, criterion)
        self.plot_results(history, cm)
        
        return history, final_metrics

    def save_model(self, model, epoch, accuracy):
        """Save model with experiment info."""
        Path(self.train_cfg.out_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.train_cfg.out_dir}/models").mkdir(exist_ok=True)
        Path(f"{self.train_cfg.out_dir}/plots").mkdir(exist_ok=True)
        
        model_path = f"{self.train_cfg.out_dir}/models/best_{self.model_cfg.model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': self.classes,
            'config': self.config.to_dict(),
            'seed': self.train_cfg.seed,
            'val_accuracy': accuracy,
            'epoch': epoch
        }, model_path)
        
        with open(f"{self.train_cfg.out_dir}/classes.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.classes))
        
        print(f"Model saved: {model_path}")

    def final_evaluation(self, model, test_loader, criterion):
        """Final evaluation on test set."""
        test_loss, test_acc, test_cm, test_report = self.validate(
            model, test_loader, criterion, self.train_cfg.device
        )
        
        print(f"\nFINAL TEST RESULTS:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(test_report)
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'confusion_matrix': test_cm,
            'report': test_report
        }

    def plot_results(self, history, cm):
        """Plot training results."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        plt.title(f'{self.model_cfg.model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        plt.title(f'{self.model_cfg.model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.train_cfg.out_dir}/plots/{self.model_cfg.model_name}_training.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved: artifacts/plots/{self.model_cfg.model_name}_training.png")


def export_onnx(model_name, num_classes, img_size=224):
    """Export model to ONNX format."""
    device = torch.device("cpu")
    
    model_path = f"artifacts/models/best_{model_name}.pth"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found for ONNX export")
        return False
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval().to(device)
        
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
        onnx_path = f"artifacts/best_{model_name}.onnx"
        
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=12
        )
        print(f"ONNX model exported: {onnx_path}")
        return True
    except Exception as e:
        print(f"ONNX export failed for {model_name}: {e}")
        return False


def get_alternative_model_config():
    """Get configuration for alternative model (MobileNetV3)."""
    return ExperimentConfig(
        data=DataConfig(batch_size=32, img_size=224),
        model=ModelConfig(
            model_name="mobilenetv3_large_100",  
            num_classes=3,
            pretrained=True,
            freeze_backbone=True,
            unfreeze_epoch=3,
            learning_rate=0.001,
            weight_decay=0.01
        ),
        train=TrainConfig(epochs=15, seed=42)
    )


def main():
    """Main function with robust error handling."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("CAR CLASSIFICATION TRAINING")
    print("Using ResNet50 + MobileNetV3 as models")
    print("=" * 55)
    
    trainer = ModelTrainer(EXPERIMENT_CONFIGS['resnet50'])
    trainer.setup_data_directories()
    
    if not trainer.check_data_exists("data/raw"):
        print("\nPlease add your images to:")
        print("   data/raw/minivan/  (30+ images)")
        print("   data/raw/sedan/    (30+ images)")
        print("   data/raw/wagon/    (30+ images)")
        sys.exit(1)
    
    models_to_try = [
        ('resnet50', EXPERIMENT_CONFIGS['resnet50']),
        ('mobilenetv3_large_100', get_alternative_model_config()),
    ]
    
    results = {}
    successful_models = []
    
    for model_name, exp_config in models_to_try:
        print(f"\n{'='*50}")
        print(f"PROCESSING: {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            # Try to train the model
            trainer = ModelTrainer(exp_config)
            history, metrics = trainer.train()
            
            results[model_name] = {
                'history': history,
                'metrics': metrics
            }
            successful_models.append(model_name)
            
            # Export to ONNX
            export_success = export_onnx(model_name, exp_config.model.num_classes)
            if export_success:
                print(f"{model_name} - Training and export completed!")
            else:
                print(f"{model_name} - Training completed but ONNX export failed")
                
        except Exception as e:
            print(f"{model_name} training failed: {e}")
            
            if model_name == 'resnet50':
                print("CRITICAL: First model failed! Stopping execution.")
                sys.exit(1)
            else:
                print("NON-CRITICAL: Second model failed. Continuing...")
                continue
    
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    if successful_models:
        print("Successfully trained models:")
        for model_name in successful_models:
            test_acc = results[model_name]['metrics']['test_acc']
            print(f" {model_name}: Test Accuracy = {test_acc:.4f}")
        
        if len(successful_models) > 1:
            best_model = max(successful_models, key=lambda x: results[x]['metrics']['test_acc'])
            best_acc = results[best_model]['metrics']['test_acc']
            print(f"\nBEST MODEL: {best_model} with accuracy {best_acc:.4f}")
    else:
        print("No models were successfully trained")
    
    print(f"\nResults saved in: artifacts/")
    print("Training completed!")


if __name__ == "__main__":
    main()