# app/app.py
import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os
from pathlib import Path
import json

# Автоматически определяем лучшую модель
def find_best_model():
    """Find the best model based on testing results."""
    # Сначала проверяем информацию о лучшей модели
    best_model_info_path = Path("../experiments/artifacts/best_model.json")
    
    if best_model_info_path.exists():
        try:
            with open(best_model_info_path, 'r', encoding='utf-8') as f:
                best_model_info = json.load(f)
            best_model_name = best_model_info['name']
            best_accuracy = best_model_info['accuracy']
            
            # Ищем ONNX версию лучшей модели в правильной папке
            onnx_path = Path(f"../experiments/artifacts/plots/best_{best_model_name}.onnx")
            if onnx_path.exists():
                print(f"🎯 Используем лучшую модель: {best_model_name} (точность: {best_accuracy:.1%})")
                return onnx_path
            else:
                print(f"⚠️ ONNX версия лучшей модели не найдена по пути: {onnx_path}")
        except Exception as e:
            print(f"⚠️ Ошибка чтения best_model.json: {e}")
    
    # Fallback: ищем любую доступную ONNX модель в правильных папках
    possible_paths = [
        Path("../experiments/artifacts/plots"),  # Где реально лежат ONNX файлы
        Path("../experiments/artifacts"),        # На всякий случай проверим корень
        Path("../experiments/artifacts/models")  # И папку models
    ]
    
    for models_dir in possible_paths:
        if models_dir.exists():
            onnx_files = list(models_dir.glob("best_*.onnx"))
            if onnx_files:
                fallback_model = onnx_files[0]
                print(f"✅ Найдена модель: {fallback_model}")
                return fallback_model
    
    # Если ничего не нашли, ищем рекурсивно во всех подпапках artifacts
    artifacts_dir = Path("../experiments/artifacts")
    onnx_files = list(artifacts_dir.rglob("best_*.onnx"))
    
    if onnx_files:
        fallback_model = onnx_files[0]
        print(f"✅ Найдена модель (рекурсивный поиск): {fallback_model}")
        return fallback_model
    
    raise FileNotFoundError(f"No ONNX models found! Searched in: {[str(p) for p in possible_paths]}")

# Загружаем классы
try:
    with open("../experiments/artifacts/classes.txt", "r", encoding="utf-8") as f:
        CLASSES = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"⚠️ Ошибка загрузки классов: {e}")
    CLASSES = ["minivan", "sedan", "wagon"]

print(f"🎯 Загружены классы: {CLASSES}")

# Загружаем модель
try:
    onnx_path = find_best_model()
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    print(f"✅ Модель загружена: {onnx_path.name}")
    print(f"📁 Полный путь: {onnx_path.absolute()}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    session = None

# Трансформации (должны совпадать с обучением)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def predict(image):
    if session is None:
        return {"Error": "Model not loaded"}
    
    try:
        # Предобработка
        image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).numpy()
        
        # Предсказание
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        ort_outs = session.run(None, ort_inputs)
        
        # Вычисление вероятностей
        probabilities = torch.softmax(torch.tensor(ort_outs[0]), dim=1)
        
        # Результаты
        results = {CLASSES[i]: float(probabilities[0][i]) for i in range(len(CLASSES))}
        return results
        
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}

# Создание интерфейса
def create_examples():
    """Create example images list"""
    examples = []
    data_dirs = ["minivan", "sedan", "wagon"]
    
    for class_name in data_dirs:
        class_path = Path(f"../data/raw/{class_name}")
        if class_path.exists():
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
            if images:
                examples.append(str(images[0]))
    
    return examples

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение автомобиля"),
    outputs=gr.Label(num_top_classes=3, label="Предсказание модели"),
    title="🚗 Классификатор типов автомобилей",
    description="Загрузите фото автомобиля, и модель определит его тип",
    examples=create_examples() if create_examples() else None
)

if __name__ == "__main__":
    print("🚀 Starting Gradio app...")
    print("📱 Open http://localhost:7860 in your browser")
    print("🌐 Or use: http://127.0.0.1:7860")
    app.launch(server_name="localhost", server_port=7860, share=False)