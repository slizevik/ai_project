# ai_project

Проект создан на основе шаблона [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

## Установка и запуск

Для управления окружением и зависимостями используется [uv](https://docs.astral.sh/uv/).
Для команды `make` в Windows используйте WSL, как указано в задании.

```bash
make setup
make check
```

`make setup` устанавливает зафиксированные в `uv.lock` пакеты и включает Git-хуки.
`make check` проверяет стиль и типы и не изменяет файлы.

Доступны также команды напрямую:

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

Чтобы создать новый проект по тому же шаблону, установите CLI изолированно:

```bash
uv tool install cookiecutter-data-science
ccds
```

## Структура проекта

```text
├── ai_project/       <- Исходный код и скрипт для отладки проверок
├── data/             <- Данные: raw, interim, processed и external
├── docs/             <- Документация
├── models/           <- Обученные модели
├── notebooks/        <- Jupyter notebooks
├── references/       <- Справочные материалы
├── reports/          <- Отчёты и графики
├── Makefile          <- Команды setup и check
├── pyproject.toml    <- Зависимости и настройки Ruff/mypy
└── uv.lock           <- Зафиксированное окружение
```
## Проверка