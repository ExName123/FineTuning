Этот проект демонстрирует процесс настройки (fine-tuning) предобученных моделей свёрточных нейронных сетей (CNN) — таких как ResNet50 и MobileNetV3 — для задачи классификации изображений автомобилей.

Структура проекта:
Fine-Tuning/
│
├── experiments/
│   ├── train.py                 Основной скрипт обучения моделей
│   ├── analysis.py              Анализ и сравнение результатов моделей
│   ├── test_predictions.py  Тестирование и визуализация предсказаний
│
│   ├── notebooks/analysis.py  Скрипты в ячейках
│
├── data/
│   ├── raw/                     Исходные изображения
│   │   ├── sedan/
│   │   ├── minivan/
│   │   └── wagon/
│
├── artifacts/
│   ├── models/                  Сохранённые модели
│   ├── plots/                   Графики обучения и сравнения
│   ├── analysis_results.json    Отчёт анализа моделей
│   ├── best_model.json          Информация о лучшей модели
│   └── stats.json               Средние и стандартные отклонения по данным
│
│
└── README.md  

Использование:
Создать окружение
python -m venv venv
Активировать
source venv/bin/activate  (или venv\Scripts\activate на Windows)

Установить зависимости
pip install -r requirements.txt

Обучение моделей
Запуск обучения обеих моделей:
python experiments/train.py
Скрипт обучает ResNet50 и MobileNetV3
спользует fine-tuning (предобученные веса из Hugging Face Hub)Сохраняет лучшие модели по метрике val_accuracy
Генерирует графики:
Loss over epochs
Accuracy over epochs
Confusion Matrix

Результаты сохраняются в:
artifacts/models/
artifacts/plots/
artifacts/logs/

Анализ моделей
После обучения можно сравнить модели по точности и сложности:
python experiments/analysis.py

Генерирует визуализацию (model_comparison.png)
Создаёт аналитический отчёт (analysis_results.json)

Тестирование и визуализация
Проверить, как лучшая модель классифицирует реальные изображения:
python experiments/test_predictions.py

Используемые технологии
PyTorch — обучение моделей
Timm — предобученные CNN
Matplotlib / Seaborn — визуализация
Pandas / NumPy — аналитика
Pillow — обработка изображений
YAML / JSON — хранение конфигураций и отчётов