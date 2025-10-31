#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch

@dataclass
class DataConfig:
   
    data_root: str = "data/raw"
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    augmentation: bool = True
    color_jitter: float = 0.15
    random_rotate: int = 10

@dataclass  
class ModelConfig:
   
    model_name: str = "resnet50"
    num_classes: int = 3
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int = 5
    
    # Hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    @classmethod
    def get_learning_rate_space(cls):
        return [0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    @classmethod
    def get_weight_decay_space(cls):
        return [0.1, 0.01, 0.001, 0.0001, 0.0]
    
    @classmethod
    def get_unfreeze_epoch_space(cls):
        return [3, 5, 7, 10]

@dataclass
class TrainConfig:
    epochs: int = 15
    early_stopping_patience: int = 5
    
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    momentum: float = 0.9
    
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_dir: str = "artifacts"
    save_checkpoints: bool = True
    log_interval: int = 1
    
    @classmethod
    def get_batch_size_space(cls):
        return [8, 16, 32, 64]
    
    @classmethod  
    def get_epochs_space(cls):
        return [10, 15, 20, 25]

@dataclass
class HyperparameterSearchConfig:
    search_type: str = "grid"
    n_trials: int = 8
    
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.0001])
    # batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    # weight_decays: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001, 0.0001])
    epochs_list: List[int] = field(default_factory=lambda: [10, 15])
    
    tuning_epochs: int = 10
    patience: int = 3

@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    
    def to_dict(self):
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'train': self.train.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            train=TrainConfig(**config_dict['train'])
        )

RESNET50_EXPERIMENT = ExperimentConfig(
    data=DataConfig(batch_size=16, img_size=224),
    model=ModelConfig(model_name="resnet50", freeze_backbone=True, unfreeze_epoch=5),
    train=TrainConfig(epochs=15, learning_rate=0.001, seed=42)
)

MOBILENET_EXPERIMENT = ExperimentConfig(
    data=DataConfig(batch_size=32, img_size=224),
    model=ModelConfig(model_name="mobilenetv3_large_100", freeze_backbone=True, unfreeze_epoch=3),
    train=TrainConfig(epochs=15, learning_rate=0.001, seed=42)
)

EXPERIMENT_CONFIGS = {
    'resnet50': RESNET50_EXPERIMENT,
    'mobilenetv3_large_100': MOBILENET_EXPERIMENT,
}