#!/usr/bin/env python3
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def load_model_results():
    print("ЗАГРУЗКА РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
    print("=" * 50)
    
    models_info = []
    model_names = ['resnet50', 'mobilenetv3_large_100']
    
    for model_name in model_names:
        try:
            checkpoint_path = f"artifacts/models/best_{model_name}.pth"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            models_info.append({
                'Model': model_name,
                'Family': 'ResNet' if 'resnet' in model_name else 'MobileNet',
                'Test Accuracy': checkpoint.get('val_accuracy', 0),
                'Epochs Trained': checkpoint.get('epoch', 0),
                'Parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values())
            })
            print(f"{model_name}: Accuracy = {checkpoint.get('val_accuracy', 0):.4f}")
        except Exception as e:
            print(f"{model_name}: {e}")
    
    return pd.DataFrame(models_info)

def plot_comparison(df):
    """Create comparison plots."""
    print("\nСОЗДАНИЕ ГРАФИКОВ СРАВНЕНИЯ")
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bars = axes[0].bar(df['Model'], df['Test Accuracy'], 
                      color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    axes[0].set_title('Сравнение точности моделей', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    bars = axes[1].bar(df['Model'], df['Parameters'] / 1e6, 
                      color=['#2ca02c', '#d62728'], alpha=0.7)
    axes[1].set_title('Количество параметров (миллионы)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Млн. параметров')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}M', ha='center', va='bottom')
    
    family_acc = df.groupby('Family')['Test Accuracy'].mean()
    bars = axes[2].bar(family_acc.index, family_acc.values,
                      color=['#9467bd', '#8c564b'], alpha=0.7)
    axes[2].set_title('Сравнение по семействам', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Average Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены: artifacts/plots/model_comparison.png")

def generate_report(df):
    print("\nАНАЛИТИЧЕСКИЙ ОТЧЕТ")
    print("=" * 50)
    
    best_model = df.loc[df['Test Accuracy'].idxmax()]
    fastest_model = df.loc[df['Parameters'].idxmin()]
    
    print("ОСНОВНЫЕ МЕТРИКИ:")
    print(f"Лучшая модель: {best_model['Model']}")
    print(f"  - Точность: {best_model['Test Accuracy']:.4f}")
    print(f"  - Параметры: {best_model['Parameters']:,}")
    print(f"  - Семейство: {best_model['Family']}")
    
    print(f"Самая легкая модель: {fastest_model['Model']}")
    print(f"  - Параметры: {fastest_model['Parameters']:,}")
    print(f"  - Точность: {fastest_model['Test Accuracy']:.4f}")
    
    print(f"\nСРАВНЕНИЕ СЕМЕЙСТВ:")
    family_stats = df.groupby('Family').agg({
        'Test Accuracy': ['mean', 'max'],
        'Parameters': 'mean'
    }).round(4)
    print(family_stats)
    
    print(f"\nВЫВОДЫ:")
    accuracy_diff = df['Test Accuracy'].max() - df['Test Accuracy'].min()
    param_ratio = df['Parameters'].max() / df['Parameters'].min()
    
    print(f"Разница в точности: {accuracy_diff:.4f}")
    print(f"Соотношение параметров: {param_ratio:.1f}x")
    
    if accuracy_diff < 0.05:
        print("Выбрать более легкую модель (разница в точности незначительна)")
    else:
        print("Выбрать более точную модель")

def save_results(df):
    results = {
        'best_model': df.loc[df['Test Accuracy'].idxmax()].to_dict(),
        'comparison': df.to_dict('records'),
        'summary': {
            'accuracy_range': [df['Test Accuracy'].min(), df['Test Accuracy'].max()],
            'parameter_range': [df['Parameters'].min(), df['Parameters'].max()],
            'models_tested': len(df)
        }
    }
    
    with open('artifacts/analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены: artifacts/analysis_results.json")

def main():
    """Main analysis function."""
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print("=" * 60)
    
    df = load_model_results()
    
    if df.empty:
        print("Нет данных для анализа")
        return
    
    plot_comparison(df)
    
    generate_report(df)
    
    save_results(df)
    
    print(f"\nАнализ завершен!")
    print(f"Проверьте файлы в artifacts/")

if __name__ == "__main__":
    main()