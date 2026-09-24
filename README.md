# Marketplace Product Classifier

Классификация карточек товаров маркетплейса по `greatgrandparent_category` на базе `cointegrated/rubert-tiny2`

## Цель проекта

Построить компактный русский text-classification pipeline и сравнить несколько вариантов обучения на одном и том же stratified split

**Seed во всём проекте: `777`.**

## Данные

Источник: `nyuuzyou/ke-products`

После удаления строк без `greatgrandparent_category`:

- 1 401 862 объекта;
- 39 классов;
- train: 1 261 675;
- validation: 70 093;
- test: 70 094.

## Эксперименты

| Модель | Вход | Loss | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|
| V1 | title + description | Cross-Entropy | **0.968242** | **0.938793** | **0.968488** |
| V2 | title + description | Weighted CE | **0.972408** | **0.952999** | 0.972413 |
| V3 | title + description + attributes | Weighted CE | **0.972594** | **0.952233** | **0.972604** |

Модель V2 выбрана по validation Macro F1

## Финальный test

- Accuracy: **0.9734**
- Macro F1: **0.9493**
- Weighted F1: **0.9734**

Test использовался только после выбора V2 и не участвовал в выборе модели

## Ноутбуки

1. `01_data_exploration.ipynb` - исследование данных, подготовка текста, split и токенизация
2. `02_baseline_and_experiments.ipynb` - V1, V2, V3 и сравнение экспериментов
3. `03_error_analysis.ipynb` - финальный test и анализ наиболее частых ошибок
4. `04_final_evaluation_and_inference.ipynb` - загрузка финальной модели и примеры inference

## Python-файлы

В `src/project_utils.py` находятся только переиспользуемые функции и общий код:

- очистка HTML;
- подготовка текста;
- расчёт метрик;
- weighted trainer;
- функция inference.

Основная исследовательская часть проекта находится в ноутбуках

## Веса модели

Веса находятся в:

```text
models/
└── v2_weighted/
```

с сохранённой финальной V2-моделью

## Установка

```bash
pip install -r requirements.txt
```

После установки ноутбуки можно открывать в Jupyter/VS Code и выполнять последовательно

Для повторного обучения в `02_baseline_and_experiments.ipynb` нужно явно установить `RETRAIN_MODELS = True` и иметь доступ к необходимому GPU/ресурсам
