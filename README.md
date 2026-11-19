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

Результаты приведены по сохранённым `metrics.json`. Для шести CNN-запусков статус `FINISHED` и соответствие run ID гиперпараметрам проверены через MLflow. Accuracy и macro-F1 указаны в долях от 0 до 1; значения округлены.

| Запуск | Модель | Learning rate | Batch size | Лучшая эпоха | Val loss | Val accuracy | Val macro-F1 | Test loss | Test accuracy | Test macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | linear | 0.001 | 128 | 3 | 1.915676 | 0.3546 | 0.347653 | 1.912713 | 0.3478 | 0.339102 |
| grid-1 | simple_cnn | 0.0003 | 64 | 5 | 1.882206 | 0.3182 | 0.285194 | 1.853700 | 0.3216 | 0.289833 |
| grid-2 | simple_cnn | 0.0003 | 128 | 5 | 1.962108 | 0.3036 | 0.272785 | 1.932580 | 0.3097 | 0.277325 |
| grid-3 | simple_cnn | 0.001 | 64 | 5 | 1.672153 | 0.3966 | 0.386843 | 1.642183 | 0.3971 | 0.387713 |
| grid-4 | simple_cnn | 0.001 | 128 | 5 | 1.760523 | 0.3528 | 0.322806 | 1.725591 | 0.3530 | 0.325308 |
| **grid-5 (выбрана)** | **simple_cnn** | **0.003** | **64** | **5** | **1.548241** | **0.4390** | **0.430398** | **1.528632** | **0.4385** | **0.429496** |
| grid-6 | simple_cnn | 0.003 | 128 | 5 | 1.606952 | 0.4338 | 0.425122 | 1.578218 | 0.4236 | 0.412541 |

Baseline `linear-seed42` обучался на 10 000 train-изображениях с seed `42` и данными `processed/cifar10/v1`. Лучшая эпоха по validation loss — 3; записанное время обучения — 27.414 секунды. MLflow run ID: `8522bd3a833f4a06a218073b612eb8a2`.

Наличие `model.pt`, `config.yaml`, `metrics.json` и `confusion_matrix.png` во вкладке Artifacts baseline подтверждено при ручной проверке. В baseline и всех шести CNN-запусках `git_commit` записан как `unknown`; требование фиксации коммита ещё необходимо исправить и проверить новым запуском.

Выбрана CNN с `lr=0.003`, `batch_size=64`, seed `42` и checkpoint эпохи 5: у неё минимальный validation loss среди сетки — 1.548241, validation accuracy — 43.90%, macro-F1 — 0.430398. Итоговые test-метрики выбранного checkpoint: loss 1.528632, accuracy 43.85%, macro-F1 0.429496; test accuracy выше baseline (34.78%) на 9.07 процентного пункта. Выбор сделан только по validation loss, а test использован для итогового сравнения.

Серия CNN использовала 10 000 train-изображений и до 5 эпох на данных `processed/cifar10/v1`; результаты относятся к этому ограниченному CPU-режиму обучения.

| Запуск | MLflow run ID | Время из metrics.json, с |
|---|---|---:|
| grid-1 | `396a55ea88624e3781a637c843099b5b` | 84.491 |
| grid-2 | `2c1d6c4bcb834f8aaf1283419d98e121` | 72.089 |
| grid-3 | `3ff1189d7fbb46c39070344173e490c2` | 64.235 |
| grid-4 | `b03298c1842243dd81f9aa26db300952` | 53.472 |
| grid-5 | `7ca979cd2def4f9a8cd72c3874a354c7` | 64.585 |
| grid-6 | `7432a8435e2f4700928ebb7122bb526f` | 62.266 |

После экспериментов остановите инфраструктуру:

```bash
make infra-down
```
