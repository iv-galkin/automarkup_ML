# 🖼️ Automatic Image Annotation ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-wrMAGPhDLT5Y0x9i1x62JAOOh6RmBss?authuser=1#scrollTo=NDMT80Q7VKG5)


## 📖 Описание проекта
Автоматизированный пайплайн для генерации аннотаций изображений с использованием **zero-shot** и **мультимодальных моделей**.  
Детекция объектов без предварительного обучения + генерация метаданных (цвет, размер, пространственные отношения) обнаруженных объектов в формате json.

---

## 🚀 Основные возможности

* 🆓 Zero-shot детекция любых объектов, упомянутых в тексте описания.
* 📝 Генерация описаний и определение цвета каждого объекта.
* 📏 Классификация размера (small, medium, large) по площади bbox.
* 📐 Пространственные отношения между объектами (above, below, left of и т.д.).
* 🔄 Гибкие эвристики: объединение пересекающихся боксов, фильтрация по confidence, учёт количества объектов.
---

## 🤖 Особенности
- **Мультимодельный подход**: YOLO-World + BLIP-2 + SAM
- **Автоматизация**: Полный цикл от изображения до JSON-аннотации
- **Поддержка**: Работа с любыми классами объектов

---
## 🛠️ Архитектура пайплайна
![Architecture](docs/results/architecture.png)
---

## 🖥️ Стек технологий

Проект использует следующие основные технологии и библиотеки:
* Язык и среда: Python 3.8+
* Генерация описаний: BLIP-2 (Salesforce/blip2-opt-2.7b)
* Детекция объектов: YOLO-world (YOLOv8), альтернативно OWL-ViT, SAM
* Библиотеки для детекции и трекинга: Ultralytics YOLOv8, Detectron2, torchvision
* Обработка изображений: OpenCV, Pillow
* Научные вычисления: NumPy, SciPy
* Работа с данными: pandas
* Метрики и валидация: scikit-learn, pycocotools

---

## 🎨 Примеры результатов

Ниже два примера работы пайплайна: изображение с пронумерованными объектами и соответствующий JSON-вывод.

### Результат 1 🖼️

![Result 1](docs/results/vis_example1.jpg)

<details>
<summary>Показать JSON для Результата 1</summary>

```json
{
  "image_name": "example1.jpg",
  "description": "A robotic arm holding a metal tool.",
  "objects": [
    { "name": "robotic arms_1", "color": "gray", "size": "medium", "description": "Gray robotic arms holding tool" },
    { "name": "tool_1",         "color": "silver","size": "small",  "description": "Silver metal tool" }
  ],
  "2d_bbox": [
    { "object": "robotic arms_1", "bbox": [30, 40, 150, 240] },
    { "object": "tool_1",         "bbox": [160, 220, 210, 270] }
  ]
}
```
</details>

### Результат 2 🖼️

![Result 2](docs/results/vis_example2.jpg)

<details>
<summary>Показать JSON для Результата 2</summary>
    
```json
{
  "image_name": "example1.jpg",
  "description": "A robotic arm holding a metal tool.",
  "objects": [
    { "name": "robotic arms_1", "color": "gray", "size": "medium", "description": "Gray robotic arms holding tool" },
    { "name": "tool_1",         "color": "silver","size": "small",  "description": "Silver metal tool" }
  ],
  "2d_bbox": [
    { "object": "robotic arms_1", "bbox": [30, 40, 150, 240] },
    { "object": "tool_1",         "bbox": [160, 220, 210, 270] }
  ]
}
```
</details>
