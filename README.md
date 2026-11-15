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

## Эксперименты

Для трекинга используется локальный MLflow. PostgreSQL хранит метаданные запусков, RustFS хранит артефакты MLflow в bucket `mlflow`, а training-контейнер подключён к той же Docker-сети. Интерфейс MLflow доступен по адресу [http://localhost:5000](http://localhost:5000).

### Запуск инфраструктуры

```bash
cp .env.example .env
make infra-up
```

`make infra-up` собирает training-образ и запускает RustFS, PostgreSQL, init-контейнер bucket и MLflow. Credentials и настройки S3 берутся из `.env`.

### Baseline

Baseline — линейный классификатор над flattened RGB-пикселями. Для отдельного запуска используйте индекс `0`:

```bash
docker compose --profile training run --rm \
  -e CONFIG=configs/grid.yaml \
  -e GRID_INDEX=0 training
```

Конфигурация baseline находится в `configs/baseline.yaml`.

### Grid

Сетка задаётся в `configs/grid.yaml` и содержит 6 комбинаций:

```yaml
grid:
  train.lr: [0.0003, 0.001, 0.003]
  train.batch_size: [64, 128]
```

Полный запуск выполняет baseline и каждую комбинацию в отдельном одноразовом контейнере:

```bash
make experiments CONFIG=configs/grid.yaml
```

Для другого набора параметров достаточно изменить YAML-конфиг. Код обучения менять не нужно. Каждый запуск получает собственное имя и записывается в MLflow.

### Модели в S3

После запуска артефакты модели сохраняются в S3:

```text
models/<experiment_name>/<run_name>/
├── model.pt
├── config.yaml
├── metrics.json
└── confusion_matrix.png
```

`metrics.json` содержит лучшую эпоху, validation-метрики, test-метрики, seed, версию данных и git commit. Существующий run не перезаписывается; для явной замены требуется флаг `--allow-overwrite`.

### Выбор конфигурации и результаты

Test используется только для итогового замера уже выбранной модели. Конфигурация выбирается исключительно по минимальному `val_loss`; test-метрики не участвуют в выборе.

После фактического запуска серии заполните таблицу значениями из MLflow и `metrics.json`:

| Запуск | Модель | Learning rate | Batch size | Лучшая эпоха | Val loss | Val accuracy | Val macro-F1 | Test loss | Test accuracy | Test macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | linear | 0.001 | 128 | — | — | — | — | — | — | — |
| grid-1 | simple_cnn | 0.0003 | 64 | — | — | — | — | — | — | — |
| grid-2 | simple_cnn | 0.0003 | 128 | — | — | — | — | — | — | — |
| grid-3 | simple_cnn | 0.001 | 64 | — | — | — | — | — | — | — |
| grid-4 | simple_cnn | 0.001 | 128 | — | — | — | — | — | — | — |
| grid-5 | simple_cnn | 0.003 | 64 | — | — | — | — | — | — | — |
| grid-6 | simple_cnn | 0.003 | 128 | — | — | — | — | — | — | — |

В итоговом выводе укажите выбранную конфигурацию, её минимальный `val_loss`, сравнение с baseline и итоговые test-метрики. После экспериментов остановите инфраструктуру:

```bash
make infra-down
```
