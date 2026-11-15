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

## Данные

Проект использует CIFAR-10 — официальный Python-архив с сайта [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). Датасет содержит 60 000 цветных изображений 32×32 в 10 классах: 50 000 исходных train и 10 000 test. Pipeline выделяет из train 45 000 изображений для обучения и 5 000 для валидации. Изображения сохраняются в PNG, а pickle-конвертация выполняется только из официального raw-архива.

### Запуск с нуля

Для запуска нужен Docker и `uv`. В Windows команды `make` запускаются из WSL.

```bash
make setup
cp .env.example .env
make s3-up
make ingest
make data
```

RustFS доступен по адресу `http://localhost:9000`, а его web-консоль — по адресу `http://localhost:9001`. Bucket создаётся автоматически init-контейнером. Данные хранятся в именованном Docker volume и переживают `docker compose down`.

`make ingest` загружает официальный архив в S3 как неизменяемый объект:

```text
raw/cifar-10-python.tar.gz
```

Повторный запуск не скачивает архив и не перезаписывает существующий объект.

`make data` скачивает raw-объект локально, конвертирует CIFAR-10, создаёт фиксированное stratified train/val-разбиение, считает статистики только по train и публикует результат в:

```text
processed/cifar10/v1/
├── images/              # 60 000 PNG-файлов
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── stats.json
└── manifest.json
```

По умолчанию используется `validation_size=5000` и `seed=42`. Для отдельных шагов доступны CLI-модули:

```bash
uv run python -m ai_project.download_raw
uv run python -m ai_project.convert
uv run python -m ai_project.split --validation-size 5000 --seed 42
uv run python -m ai_project.stats
uv run python -m ai_project.upload_processed --version v1
```

Существующая processed-версия не перезаписывается. Для нового разбиения укажите другой seed или размер validation и новую версию.
