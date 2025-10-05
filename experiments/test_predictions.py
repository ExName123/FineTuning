#!/usr/bin/env python3
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import timm
import json

def evaluate_model_accuracy(model, model_name, classes, sample_images):
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    for true_class, img_path in sample_images:
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((224, 224))
            
            tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = torch.argmax(probabilities).item()
                predicted_class = classes[predicted_idx]
                confidence = torch.max(probabilities).item()
            
            is_correct = (predicted_class.lower() == true_class.lower())
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            results.append({
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            status = "+" if is_correct else "-"
            print(f"   {status} {true_class:10} -> {predicted_class:10} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"Ошибка с {img_path.name}: {e}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, results

def test_model_predictions():
    print("ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЙ МОДЕЛЕЙ")
    print("=" * 50)
    
    data_dir = Path("data/raw")
    sample_images = []
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            if images:
                sample_images.append((class_dir.name, images[0]))
    
    if not sample_images:
        print("Не найдены тестовые изображения")
        return None
    
    print("📸 Тестовые изображения:")
    for class_name, img_path in sample_images:
        print(f"   {class_name}: {img_path.name}")
    
    try:
        with open("artifacts/classes.txt", "r", encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"Классы: {classes}")
    except:
        classes = ["minivan", "sedan", "wagon"]
        print("Используем классы по умолчанию")
    
    model_names = ['resnet50', 'mobilenetv3_large_100']
    model_performance = {}
    
    for model_name in model_names:
        print(f"\nТЕСТИРУЕМ МОДЕЛЬ: {model_name}")
        print("-" * 40)
        
        try:
            model_path = f"artifacts/models/best_{model_name}.pth"
            checkpoint = torch.load(model_path, map_location='cpu')
            
            model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Модель загружена успешно")
            
            accuracy, results = evaluate_model_accuracy(model, model_name, classes, sample_images)
            model_performance[model_name] = {
                'accuracy': accuracy,
                'results': results
            }
            
            print(f"Точность: {accuracy:.1%} ({int(accuracy * len(sample_images))}/{len(sample_images)})")
                    
        except Exception as e:
            print(f"Ошибка загрузки модели {model_name}: {e}")
    
    if model_performance:
        best_model = max(model_performance.items(), key=lambda x: x[1]['accuracy'])
        best_model_name = best_model[0]
        best_accuracy = best_model[1]['accuracy']
        
        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name} (точность: {best_accuracy:.1%})")
        
        best_model_info = {
            'name': best_model_name,
            'accuracy': best_accuracy,
            'test_samples': len(sample_images),
            'timestamp': str(np.datetime64('now'))
        }
        
        with open('artifacts/best_model.json', 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)
        
        print(f"Информация о лучшей модели сохранена: artifacts/best_model.json")
        return best_model_name
    else:
        print("Не удалось протестировать ни одну модель")
        return None

def display_sample_predictions(best_model_name):
    print(f"\nВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ ЛУЧШЕЙ МОДЕЛИ: {best_model_name}")
    print("=" * 60)
    
    try:
        model_path = f"artifacts/models/best_{best_model_name}.pth"
        checkpoint = torch.load(model_path, map_location='cpu')
        
        with open("artifacts/classes.txt", "r", encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
        
        model = timm.create_model(best_model_name, pretrained=False, num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        data_dir = Path("data/raw")
        sample_data = []
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
                if images:
                    sample_data.append((class_dir.name, images[0]))
        
        n_images = len(sample_data)
        if n_images > 0:
            fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
            if n_images == 1:
                axes = [axes]
            
            for idx, (true_class, img_path) in enumerate(sample_data):
                if idx >= n_images:
                    break
                    
                image = Image.open(img_path).convert('RGB')
                display_image = image.resize((200, 200))
                
                tensor = torch.tensor(np.array(image.resize((224, 224)))).permute(2, 0, 1).float() / 255.0
                tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = tensor.unsqueeze(0)
                
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_idx = torch.argmax(probabilities).item()
                    predicted_class = classes[predicted_idx]
                    confidence = torch.max(probabilities).item()
                
                axes[idx].imshow(display_image)
                axes[idx].set_title(f"True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.3f}", 
                                  fontsize=12)
                axes[idx].axis('off')
                
                color = 'green' if predicted_class.lower() == true_class.lower() else 'red'
                for spine in axes[idx].spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(3)
            
            plt.tight_layout()
            plt.savefig('artifacts/plots/sample_predictions.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Визуализация сохранена: artifacts/plots/sample_predictions.png")
        else:
            print("Нет изображений для визуализации")
        
    except Exception as e:
        print(f"Ошибка визуализации: {e}")

if __name__ == "__main__":
    best_model = test_model_predictions()

    if best_model:
        display_sample_predictions(best_model)
    
    print(f"\nТЕСТИРОВАНИЕ ЗАВЕРШЕНО")