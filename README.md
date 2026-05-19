# MFDP — My First Data Project

AI-система оценки финансового риска (Home Credit Default Risk).

## Документы

- `business_understanding.md` — бизнес-анализ и прототип
- `data_understanding.md` — Data Understanding (датасет, EDA, валидация)
- `benchmarking_hw3.md` — бенчмаркинг

## Данные

CSV в Git не хранятся (DVC). После клонирования:

```bash
pip install -r requirements.txt
dvc pull
```

Если `dvc pull` недоступен: `python download.py` (нужен аккаунт Kaggle).

## Ветки

- `hw4` — Data Understanding
