# 🖼️ Automatic Image Annotation Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-link-here)

Автоматизированный пайплайн для генерации аннотаций изображений с использованием **zero-shot моделей**.  
Детекция объектов без предварительного обучения + генерация метаданных (цвет, размер, пространственные отношения).

---

## 🚀 Особенности
- **Мультимодельный подход**: YOLO-World + BLIP-2 + SAM
- **Автоматизация**: Полный цикл от изображения до JSON-аннотации
- **Оптимизация**: Фильтрация шума, каскадная обработка
- **Поддержка**: Работа с любыми классами объектов

---
## 🛠 Архитектура пайплайна
```mermaid
graph TD
    A[Входное изображение] --> B(BLIP-2: Генерация описания)
    B --> C{SpaCy: Извлечение объектов}
    C --> D[Список классов]
    D --> E(YOLO-World: Детекция)
    E --> F{Фильтрация боксов}
    F -->|NMS + Эвристики| G[Чистые детекции]
    G --> H[Генерация метаданных]
    H --> I[JSON-аннотация]
---
## 🛠 Технологии
| Категория       | Технологии                          |
|-----------------|-------------------------------------|
| **Детекция**    | YOLO-World, SAM                     |
| **Анализ**      | BLIP-2, SpaCy                       |
| **Обработка**   | OpenCV, Pillow, NumPy               |
| **Инфраструктура** | Google Colab, CUDA               |

---

## ⚡ Быстрый старт
```bash
# Установка зависимостей
pip install ultralytics transformers segment-anything spacy
python -m spacy download en_core_web_sm

# Запуск для папки с изображениями
python pipeline.py \
  --input data/raw_images \
  --output annotations
