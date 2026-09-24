# Структура проекта

```text
marketplace-product-classifier/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_and_experiments.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_final_evaluation_and_inference.ipynb
├── src/
│   └── project_utils.py
├── configs/
│   └── config.yaml
├── models/
│   └── .gitkeep
├── tests/
│   └── test_project_utils.py
├── README.md
├── requirements.txt
└── .gitignore
```

Основной артефакт проекта - ноутбуки. `src/project_utils.py` содержит только переиспользуемые функции, которые не стоит дублировать в каждом notebook.
