#!/usr/bin/env python3
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_model_results():
    print("UPLOADING MODEL RESULTS")
    
    models_info = []
    model_names = ['resnet50', 'mobilenetv3_large_100']
    
    for model_name in model_names:
        try:
            checkpoint_path = f"artifacts/models/best_{model_name}.pth"
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except:
                print(f"Warning: Using weights_only=False for {model_name}. Load from trusted source only.")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            val_accuracy = checkpoint.get('val_accuracy', 0)
            if not val_accuracy:
                val_accuracy = checkpoint.get('best_acc', checkpoint.get('accuracy', 0))
            
            epochs_trained = checkpoint.get('epoch', checkpoint.get('epochs', 0))
            
            model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
            if model_state_dict:
                parameters = sum(p.numel() for p in model_state_dict.values())
            else:
                parameters = 0
            
            models_info.append({
                'Model': model_name,
                'Family': 'ResNet' if 'resnet' in model_name else 'MobileNet',
                'Test Accuracy': float(val_accuracy),
                'Epochs Trained': int(epochs_trained),
                'Parameters': int(parameters)
            })
            print(f"{model_name}: Accuracy = {val_accuracy:.4f}, Parameters = {parameters:,}")
            
        except Exception as e:
            print(f"{model_name}: {e}")
    
    return pd.DataFrame(models_info)

def load_results_from_csv():
    print("\nTRYING TO LOAD FROM CSV FILES...")
    
    models_info = []
    csv_files = {
        'resnet50': 'resnet50_tuning_results.csv',
        'mobilenetv3_large_100': 'mobilenetv3_large_100_tuning_results.csv'
    }
    
    for model_name, csv_file in csv_files.items():
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if not df.empty:
                    best_result = df.loc[df['val_acc'].idxmax()]
                    
                    models_info.append({
                        'Model': model_name,
                        'Family': 'ResNet' if 'resnet' in model_name else 'MobileNet',
                        'Test Accuracy': float(best_result['val_acc']),
                        'Epochs Trained': int(best_result.get('epochs', best_result.get('best_epoch', 0))),
                        'Parameters': int(estimate_parameters(model_name))
                    })
                    print(f"{model_name} from CSV: Accuracy = {best_result['val_acc']:.4f}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return pd.DataFrame(models_info)

def estimate_parameters(model_name):
    param_estimates = {
        'resnet50': 25_557_032,
        'mobilenetv3_large_100': 5_483_032
    }
    return param_estimates.get(model_name, 0)

def plot_comparison(df):
    print("\nCREATING COMPARISON CHARTS")
    
    os.makedirs('artifacts/plots', exist_ok=True)
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bars = axes[0].bar(df['Model'], df['Test Accuracy'], 
                      color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    if df['Parameters'].sum() > 0:
        bars = axes[1].bar(df['Model'], df['Parameters'] / 1e6, 
                          color=['#2ca02c', '#d62728'], alpha=0.7)
        axes[1].set_title('Number of Parameters (Millions)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Million Parameters')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}M', ha='center', va='bottom', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Parameter data\nnot available', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Number of Parameters', fontsize=14, fontweight='bold')
    
    family_acc = df.groupby('Family')['Test Accuracy'].mean()
    if len(family_acc) > 1:
        bars = axes[2].bar(family_acc.index, family_acc.values,
                          color=['#9467bd', '#8c564b'], alpha=0.7)
        axes[2].set_title('Accuracy by Model Family', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Average Accuracy')
        axes[2].set_ylim(0, 1)
        axes[2].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'Only one model family\navailable', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Accuracy by Model Family', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graphs saved: artifacts/plots/model_comparison.png")

def generate_report(df):
    print("\nANALYTICAL REPORT")
    
    if df.empty:
        print("No data available for analysis")
        return
    
    best_model = df.loc[df['Test Accuracy'].idxmax()]
    
    print("BASIC METRICS:")
    print(f"Best Model: {best_model['Model']}")
    print(f"  - Accuracy: {best_model['Test Accuracy']:.4f} ({best_model['Test Accuracy']*100:.2f}%)")
    print(f"  - Parameters: {best_model['Parameters']:,}")
    print(f"  - Family: {best_model['Family']}")
    print(f"  - Epochs Trained: {best_model['Epochs Trained']}")
    
    if len(df) > 1:
        fastest_model = df.loc[df['Parameters'].idxmin()]
        print(f"\nLightest Model: {fastest_model['Model']}")
        print(f"  - Parameters: {fastest_model['Parameters']:,}")
        print(f"  - Accuracy: {fastest_model['Test Accuracy']:.4f}")
    
    print(f"\nFAMILY COMPARISON:")
    family_stats = df.groupby('Family').agg({
        'Test Accuracy': ['mean', 'max', 'count'],
        'Parameters': 'mean'
    }).round(4)
    print(family_stats)
    
    print(f"\nKEY INSIGHTS:")
    if len(df) > 1:
        accuracy_diff = float(df['Test Accuracy'].max() - df['Test Accuracy'].min())
        if df['Parameters'].sum() > 0:
            param_ratio = float(df['Parameters'].max() / df['Parameters'].min())
            print(f"Accuracy difference: {accuracy_diff:.4f}")
            print(f"Parameter ratio: {param_ratio:.1f}x")
            
            if accuracy_diff < 0.05:
                print("Choose the lighter model (accuracy difference is negligible)")
            else:
                print("Choose the more accurate model")
        else:
            print(f"Accuracy difference: {accuracy_diff:.4f}")
            print("RECOMMENDATION: Choose model with highest accuracy")
    else:
        print(f"Only one model available: {best_model['Model']}")
        print(f"Final accuracy: {best_model['Test Accuracy']:.4f}")

def save_results(df):
    os.makedirs('artifacts', exist_ok=True)
    
    if df.empty:
        results = {'error': 'No data available for analysis'}
    else:
        best_model = df.loc[df['Test Accuracy'].idxmax()]
        
        best_model_dict = {}
        for key, value in best_model.items():
            if pd.isna(value):
                best_model_dict[key] = None
            elif isinstance(value, (np.integer, pd.Int64Dtype)):
                best_model_dict[key] = int(value)
            elif isinstance(value, (np.floating, pd.Float64Dtype)):
                best_model_dict[key] = float(value)
            else:
                best_model_dict[key] = value
        
        comparison_list = []
        for _, row in df.iterrows():
            row_dict = {}
            for key, value in row.items():
                if pd.isna(value):
                    row_dict[key] = None
                elif isinstance(value, (np.integer, pd.Int64Dtype)):
                    row_dict[key] = int(value)
                elif isinstance(value, (np.floating, pd.Float64Dtype)):
                    row_dict[key] = float(value)
                else:
                    row_dict[key] = value
            comparison_list.append(row_dict)
        
        results = {
            'best_model': best_model_dict,
            'comparison': comparison_list,
            'summary': {
                'accuracy_range': [
                    float(df['Test Accuracy'].min()), 
                    float(df['Test Accuracy'].max())
                ],
                'parameter_range': [
                    int(df['Parameters'].min()), 
                    int(df['Parameters'].max())
                ],
                'models_tested': int(len(df)),
                'best_accuracy': float(best_model['Test Accuracy'])
            }
        }
    
    with open('artifacts/analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved: artifacts/analysis_results.json")

def main():
    print("ANALYSIS OF EXPERIMENTAL RESULTS")
    
    df = load_model_results()
    
    if df.empty:
        print("\nTrying alternative data sources...")
        df = load_results_from_csv()
    
    if df.empty:
        print("\nNo data available for analysis")
        print("Please check:")
        print("1. Model files exist in artifacts/models/")
        print("2. CSV files with results exist")
        return
    
    plot_comparison(df)
    
    generate_report(df)
    
    save_results(df)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Check files in artifacts/ directory")

if __name__ == "__main__":
    main()