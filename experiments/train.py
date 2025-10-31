#!/usr/bin/env python3
"""
Main training script with robust error handling for model downloads.
Uses ResNet50 as first model and MobileNetV3 as second model.
"""

import os
import random
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import warnings

Path("artifacts").mkdir(exist_ok=True)
log_file = "artifacts/training.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

warnings.filterwarnings('ignore')
os.environ['TIMM_DOWNLOAD_TIMEOUT'] = '30'
os.environ['TIMM_DOWNLOAD_RETRY'] = '2'

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import argparse       
import pandas as pd    
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from config import DataConfig, HyperparameterSearchConfig, ModelConfig, TrainConfig, ExperimentConfig, EXPERIMENT_CONFIGS


class ModelTrainer:
    def __init__(self, experiment_config: ExperimentConfig):
        stats_path = "artifacts/stats.json"
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"{stats_path} не найден.")
    
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.DATA_MEAN = stats["mean"]
        self.DATA_STD = stats["std"]

        self.config = experiment_config
        self.data_cfg = experiment_config.data
        self.model_cfg = experiment_config.model
        self.train_cfg = experiment_config.train
        self.set_seed()
        
    def set_seed(self):
        random.seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)
        torch.manual_seed(self.train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.train_cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def setup_data_directories(self):
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
            print(f"Directory {data_root} does not exist")
            logger.info(f"Directory {data_root} does not exist")
            return False
            
        subdirs = [d for d in Path(data_root).iterdir() if d.is_dir()]
        if not subdirs:
            print(f"No class directories in {data_root}!")
            logger.info(f"No class directories in {data_root}!")
            return False
            
        print(f"Data found in: {data_root}")
        logger.info(f"Data found in: {data_root}")
        total_images = 0
        
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
            images = [f for f in subdir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
            
            print(f" {subdir.name}: {len(images)} images")
            logger.info(f" {subdir.name}: {len(images)} images")
            total_images += len(images)
            
        print(f"Total images: {total_images}")
        logger.info(f"Total images: {total_images}")
        
        if total_images < 30:
            print("Less than 30 images total - may affect model performance")
            logger.info("Less than 30 images total - may affect model performance")
            
        return total_images > 0
        
    def get_transforms(self):
        if self.data_cfg.augmentation:
            train_t = transforms.Compose([
                transforms.RandomResizedCrop(self.data_cfg.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(self.DATA_MEAN, self.DATA_STD)
            ])
        else:
            train_t = transforms.Compose([
                transforms.Resize((self.data_cfg.img_size, self.data_cfg.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.DATA_MEAN, self.DATA_STD)
            ])
            
        val_t = transforms.Compose([
            transforms.Resize((self.data_cfg.img_size, self.data_cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.DATA_MEAN, self.DATA_STD)
        ])
        
        return train_t, val_t

    def create_data_loaders(self):
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
        logger.info(f"Classes: {self.classes}")
        print(f"Dataset split: train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        logger.info(f"Dataset split: train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        return train_loader, val_loader, test_loader

    def create_model(self):
        print(f"Creating model: {self.model_cfg.model_name}")
        logger.info(f"Creating model: {self.model_cfg.model_name}")
        print(f"Pretrained: {self.model_cfg.pretrained}")
        logger.info(f"Pretrained: {self.model_cfg.pretrained}")
        
        try:
            model = timm.create_model(
                self.model_cfg.model_name, 
                pretrained=self.model_cfg.pretrained, 
                num_classes=self.model_cfg.num_classes
            )
            print(f"Successfully created {self.model_cfg.model_name}")
            logger.info(f"Successfully created {self.model_cfg.model_name}")
            
        except Exception as e:
            print(f"Failed to download {self.model_cfg.model_name}: {e}")
            logger.info(f"Failed to download {self.model_cfg.model_name}: {e}")
            print("Falling back to randomly initialized model...")
            logger.info("Falling back to randomly initialized model...")
            
            model = timm.create_model(
                self.model_cfg.model_name, 
                pretrained=False,  
                num_classes=self.model_cfg.num_classes
            )
            print(f"Created {self.model_cfg.model_name} with random initialization")
            logger.info(f"Created {self.model_cfg.model_name} with random initialization")
        
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
            logger.info(f"Freezing: {frozen_count} frozen, {trainable_count} trainable")
            print(f"Unfreeze at epoch: {self.model_cfg.unfreeze_epoch}")
            logger.info(f"Unfreeze at epoch: {self.model_cfg.unfreeze_epoch}")
                    
        return model

    def unfreeze_model(self, model):
        print("Unfreezing all parameters")
        logger.info("Unfreezing all parameters")
        for param in model.parameters():
            param.requires_grad = True

    def train_epoch(self, model, loader, criterion, optimizer, device):
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
        logger.info(f"\nTraining {self.model_cfg.model_name}")
        
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        model = self.create_model()
        model = model.to(self.train_cfg.device)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
        logger.info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
        
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
        logger.info(f"\nTraining for {self.train_cfg.epochs} epochs on {self.train_cfg.device}")
        
        for epoch in range(self.train_cfg.epochs):
            if epoch == self.model_cfg.unfreeze_epoch and self.model_cfg.freeze_backbone:
                print(f"\nEpoch {epoch}: unfreezing backbone")
                logger.info(f"\nEpoch {epoch}: unfreezing backbone")
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
                      f"Train Loss: {train_loss:.4f}, accuracy: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
                logger.info(f"Epoch {epoch+1}/{self.train_cfg.epochs}: "
                            f"Train Loss: {train_loss:.4f}, accuracy: {train_acc:.4f} | "
                            f"Val Loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(model, epoch, val_acc)
                if (epoch + 1) % self.train_cfg.log_interval == 0:
                    print(f"New best model Val accuracy: {val_acc:.4f}")
                    logger.info(f"New best model Val accuracy: {val_acc:.4f}")
        
        final_metrics = self.final_evaluation(model, test_loader, criterion)
        self.plot_results(history, cm)
        
        return history, final_metrics

    def save_model(self, model, epoch, accuracy):
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
        logger.info(f"Model saved: {model_path}")

    def final_evaluation(self, model, test_loader, criterion):
        """Final evaluation on test set."""
        test_loss, test_acc, test_cm, test_report = self.validate(
            model, test_loader, criterion, self.train_cfg.device
        )
        
        print(f"\nFinal test result:")
        logger.info(f"\nFinal test result:")
        print(f"   Test loss: {test_loss:.4f}")
        logger.info(f"   Test loss: {test_loss:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}")
        logger.info(f"   Test accuracy: {test_acc:.4f}")
        print("\nClassification report:")
        logger.info("\nClassification report:")
        print(test_report)
        logger.info(test_report)
        
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'confusion_matrix': test_cm,
            'report': test_report
        }

    def plot_results(self, history, cm):
        Path(f"{self.train_cfg.out_dir}/plots").mkdir(parents=True, exist_ok=True)

        best_epoch = int(np.argmax(history['val_acc']))
        best_val_acc = history['val_acc'][best_epoch]
        best_lr = history['learning_rates'][best_epoch] if 'learning_rates' in history else None

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        axes[0].plot(history['train_loss'], label='Train loss', linewidth=2, color='tab:blue')
        axes[0].plot(history['val_loss'], label='Validation loss', linewidth=2, color='tab:orange')
        axes[0].axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
        axes[0].set_title('Graph of the loss function (loss)', fontsize=12)
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Value Loss', fontsize=10)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['train_acc'], label='Train accuracy', linewidth=2, color='tab:green')
        axes[1].plot(history['val_acc'], label='Validation accuracy', linewidth=2, color='tab:red')
        axes[1].axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
        axes[1].annotate(
            f"Best epoch: {best_epoch + 1}\nVal acc = {best_val_acc:.4f}\nLR = {best_lr:.6f}",
            xy=(best_epoch, best_val_acc),
            xytext=(best_epoch + 0.5, best_val_acc - 0.05),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        axes[1].set_title('Graph accuracy', fontsize=12)
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('The proportion of correct answers', fontsize=10)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                    xticklabels=self.classes, yticklabels=self.classes)
        axes[2].set_xlabel('Predicted class', fontsize=10)
        axes[2].set_ylabel('The true class', fontsize=10)
        axes[2].set_title('Confusion Matrix', fontsize=12)

        summary_text = (
            f"Best epoch: {best_epoch + 1} | "
            f"Val acc = {best_val_acc:.4f}"
            + (f" | Learning rate = {best_lr:.6f}" if best_lr is not None else "")
        )
        fig.suptitle(summary_text, fontsize=13, fontweight='bold', y=1.02)

        plt.tight_layout()
        plot_path = f"{self.train_cfg.out_dir}/plots/{self.model_cfg.model_name}_training.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plots (loss/acc/confusion) saved: {plot_path}")
        logger.info(f"Plots (loss/acc/confusion) saved: {plot_path}")

    @staticmethod
    def export_onnx(model_name, num_classes, img_size=224):
        device = torch.device("cpu")
        model_path = f"artifacts/models/best_{model_name}.pth"
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found for ONNX export")
            logger.info(f"Model {model_path} not found for ONNX export")
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
            logger.info(f"ONNX model exported: {onnx_path}")
            return True
        except Exception as e:
            print(f"ONNX export failed for {model_name}: {e}")
            logger.info(f"ONNX export failed for {model_name}: {e}")
            return False


def get_alternative_model_config():
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\nCar classification training")
    logger.info("\nCar classification training")
    print("Using ResNet50 + MobileNetV3 as models")
    logger.info("Using ResNet50 + MobileNetV3 as models")

    trainer = ModelTrainer(EXPERIMENT_CONFIGS['resnet50'])
    trainer.setup_data_directories()

    if not trainer.check_data_exists("data/raw"):
        print("\nPlease add your images to:")
        logger.info("\nPlease add your images to:")
        print("   data/raw/minivan/  (30+ images)")
        logger.info("   data/raw/minivan/  (30+ images)")
        print("   data/raw/sedan/    (30+ images)")
        logger.info("   data/raw/sedan/    (30+ images)")
        print("   data/raw/wagon/    (30+ images)")
        logger.info("   data/raw/wagon/    (30+ images)")
        sys.exit(1)

    models_to_try = [
        ('resnet50', EXPERIMENT_CONFIGS['resnet50']),
        ('mobilenetv3_large_100', get_alternative_model_config()),
    ]

    if args.tune:
        hpcfg = HyperparameterSearchConfig()
        for model_name, exp_cfg in models_to_try:
            print(f"\nHyperparameter tunning for {model_name.upper()}")
            logger.info(f"\nHyperparameter tunning for {model_name.upper()}")
            results = []

            for lr in hpcfg.learning_rates:
                for ep in hpcfg.epochs_list:
                    print(f"\nTraining {model_name} with lr={lr}, epochs={ep}")
                    logger.info(f"\nTraining {model_name} with lr={lr}, epochs={ep}")

                    exp_cfg.model.learning_rate = lr
                    exp_cfg.train.epochs = ep

                    trainer = ModelTrainer(exp_cfg)
                    try:
                        history, metrics = trainer.train()
                        val_acc = metrics['test_acc']
                        val_cm = metrics['confusion_matrix']

                        cm_path = f"artifacts/plots/{model_name}_lr{lr}_ep{ep}_cm.png"
                        plt.figure(figsize=(5, 5))
                        sns.heatmap(
                            val_cm,
                            annot=True,
                            fmt='d',
                            cmap='Blues',
                            xticklabels=trainer.classes,
                            yticklabels=trainer.classes
                        )
                        plt.title(f"{model_name} | lr={lr}, epochs={ep}")
                        plt.xlabel("Predicted")
                        plt.ylabel("True")
                        plt.tight_layout()
                        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Confusion matrix saved: {cm_path}")
                        logger.info(f"Confusion matrix saved: {cm_path}")

                    except Exception as e:
                        print(f"Training failed for {model_name} (lr={lr}, ep={ep}): {e}")
                        logger.info(f"Training failed for {model_name} (lr={lr}, ep={ep}): {e}")
                        val_acc = 0.0

                    results.append({
                        'model': model_name,
                        'lr': lr,
                        'epochs': ep,
                        'val_acc': val_acc
                    })

            df = pd.DataFrame(results)
            csv_path = f"artifacts/{model_name}_tuning_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved results: {csv_path}")
            logger.info(f"Saved results: {csv_path}")

            best_row = df.loc[df['val_acc'].idxmax()]
            best_lr = best_row['lr']
            best_ep = int(best_row['epochs'])
            print(f"\nBest for {model_name}: lr={best_lr}, epochs={best_ep}, acc={best_row['val_acc']:.4f}")
            logger.info(f"\nBest for {model_name}: lr={best_lr}, epochs={best_ep}, acc={best_row['val_acc']:.4f}")

            exp_cfg.model.learning_rate = best_lr
            exp_cfg.train.epochs = best_ep
            print(f"Retraining {model_name} with best params...")
            logger.info(f"Retraining {model_name} with best params...")
            trainer = ModelTrainer(exp_cfg)
            trainer.train()

    else:
        results = {}
        successful_models = []

        for model_name, exp_config in models_to_try:
            print(f"Processing model: {model_name.upper()}")
            logger.info(f"Processing model: {model_name.upper()}")

            try:
                trainer = ModelTrainer(exp_config)
                history, metrics = trainer.train()

                results[model_name] = {
                    'history': history,
                    'metrics': metrics
                }
                successful_models.append(model_name)

                export_success = ModelTrainer.export_onnx(model_name, exp_config.model.num_classes)
                if export_success:
                    print(f"{model_name} - Training and export completed")
                    logger.info(f"{model_name} - Training and export completed")
                else:
                    print(f"{model_name} - Training completed but ONNX export failed")
                    logger.info(f"{model_name} - Training completed but ONNX export failed")

            except Exception as e:
                print(f"{model_name} training failed: {e}")
                logger.info(f"{model_name} training failed: {e}")
                if model_name == 'resnet50':
                    print("First model failed. Stopping execution.")
                    logger.info("First model failed. Stopping execution.")
                    sys.exit(1)
                else:
                    print("continuing...")
                    logger.info("continuing...")
                    continue

        print("Summary")
        logger.info("Summary")

        if successful_models:
            print("Successfully trained models:")
            logger.info("Successfully trained models:")
            for model_name in successful_models:
                test_acc = results[model_name]['metrics']['test_acc']
                print(f"{model_name}: Test accuracy = {test_acc:.4f}")
                logger.info(f"{model_name}: Test accuracy = {test_acc:.4f}")

            if len(successful_models) > 1:
                best_model = max(successful_models, key=lambda x: results[x]['metrics']['test_acc'])
                best_acc = results[best_model]['metrics']['test_acc']
                print(f"\nBest model: {best_model} with accuracy {best_acc:.4f}")
                logger.info(f"\nBest model: {best_model} with accuracy {best_acc:.4f}")
        else:
            print("No models were successfully trained")
            logger.info("No models were successfully trained")

        print(f"\nResults saved in: artifacts/")
        logger.info(f"\nResults saved in: artifacts/")
        print("Training completed successfully!")
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
