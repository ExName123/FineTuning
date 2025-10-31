#!/usr/bin/env python3
"""
Compute dataset statistics: mean and std per RGB channel.
Save results to artifacts/stats.json for later use in normalization.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import json

def main():
    data_dir = Path("data/raw")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # количество изображений в батче
        images = images.view(batch_samples, images.size(1), -1)  # B x C x (H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    mean = mean.tolist()
    std = std.tolist()

    print(f"\nDataset statistics:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    stats_path = Path("artifacts/stats.json")
    with stats_path.open("w") as f:
        json.dump({"mean": mean, "std": std}, f)

    print(f"\nStatistics saved to {stats_path}")

if __name__ == "__main__":
    main()
