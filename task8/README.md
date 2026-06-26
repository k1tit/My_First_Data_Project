# Домашнее задание №8 — MVP сервиса

**Продукт:** AI-система оценки кредитного риска (Home Credit).  
**Модель:** [HW7](../notebooks/hw7_improved_model.ipynb).

---

## Задание (урок 8)

| № | Требование | Баллы |
|---|------------|-------|
| 1 | Спроектировать доменную модель сервиса | 2 |
| 2 | Обеспечить хранение данных за счёт СУБД | 2 |
| 3 | Реализовать REST интерфейс | 2 |
| 4 | Реализовать пользовательский интерфейс | 3 |
| 5 | Покрытие тестами критических частей | 2 |
| 6 | Docker контейнер | 2 |
| 7 | Масштабирование воркеров с моделью | 2 |

---

## Подготовка (один раз)

```bash
# из корня репозитория MFDP — экспорт модели HW7
python task8/scripts/export_artifacts.py

cd task8
cp .env.example .env
docker compose up --build
```

| URL | Назначение |
|-----|------------|
| http://localhost:8082 | UI — форма заявки |
| http://localhost:8082/docs | REST Swagger |
| http://localhost:15673 | RabbitMQ UI |

**Масштаб воркеров:** `docker compose up --scale worker=3`

**Тесты:** `cd task8 && pip install -r app/requirements.txt -r tests/requirements.txt && PYTHONPATH=app:shared pytest tests/ -v`

---

## Где смотреть каждый пункт

| Пункт | Путь |
|-------|------|
| 1 | `docs/domain.md`, `shared/models.py` |
| 2 | `docker-compose.yml` → `database`, `shared/db.py` |
| 3 | `app/routes/score.py`, `/api/v1/score` |
| 4 | `app/templates/`, `app/routes/ui.py` |
| 5 | `tests/test_api.py` |
| 6 | `docker-compose.yml`, `app/Dockerfile`, `worker/Dockerfile` |
| 7 | `worker/worker.py`, `--scale worker=N` |

---

## Статус

- [x] 1. Доменная модель
- [x] 2. PostgreSQL
- [x] 3. REST API
- [x] 4. Web UI
- [x] 5. Тесты
- [x] 6. Docker
- [x] 7. Воркеры + scale
