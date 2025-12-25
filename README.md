# ⛅️ Классификация изображений погоды

## Постановка задачи

Задача: создать модель CV, которая бы классифицировала снимки на 11 классов
(туман, гроза, и т.д.). Модель можно использовать в сервисах погоды, городских
службах (например, для регулирования освещения), сельском хозяйстве и
транспорте.

## Формат входных и выходных данных

Вход: изображение формата .jpg. Цветовое пространство: RGB. Изображения могут
быть разной размерности. Выход: строка с названием погодного явления.

## Метрики

Для оценки модели будем использовать следующие метрики:

**Accuracy** (целевой показатель ≥85%) - показывает общую долю правильных
предсказаний.

**Macro F1-score** (целевой показатель ≥80%) - учитывает дисбаланс классов и
важен для равнозначности всех типов погоды.

Выбор целевых показателей связан с тем, что нам необходимо получить высокое
качество предсказаний (>= 80%), но достижимое для компактных моделей (ResNet18),
которые быстро развертываются и масштабируются.

## Валидация и тест

Разобьем на обучающую и тестовую выборки в соотношении 80/20. Для валидации
модели во время обучения выделим на валидационную выборку 20% от обучающей. Для
воспроизводимости зафиксируем seed = 42 (этот параметр можно задать при запуске
обучения в Hydra CLI).

## Датасеты

Будем использовать датасет с
[kaggle-соревнования](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset).
В нем представлено суммарно 6862 изображений, распределенных по 11 каталогам
(отдельный каталог создан для каждого класса). Суммарный объем файлов составляет
630 Мб. Все изображения имеют формат .jpg, однако у них нет единой размерности.
Чтобы избежать ошибок при моделировании, необходимо привести изображения к
единому размеру. Также ряд пользователей указывали на наличие нескольких
изображений, не соответствующих формату .jpg. Чтобы избежать ошибок, добавим
обработку исключений.

## Моделирование

### Бейзлайн

В качестве baseline будем использовать простую CNN с 3 сверточными слоями.

### Основная модель

В качестве основной модели будем использовать resnet18, предобученную на
датасете ImageNet. Это сверточная нейронная сеть с 18 слоями (17 сверточных и 1
полносвязный), использующая остаточные соединения для решения проблемы затухания
градиента. Адаптировать модель к новой задаче будем с помощью Fine-tuning.

### Гиперпараметры обучения

**Оптимизатор**: Adam + ReduceLROnPlateau

**Learning rate**: 1e-3, (по умолчанию)

**weight_decay:** 1e-4 (по умолчанию)

**Loss**: CrossEntropyLoss

**Эпохи**: 10 (по умолчанию)

**Batch size**: 512 (по умолчанию)

**Callbacks**: EarlyStopping (patience=5), ModelCheckpoint (best val_loss)

**Фреймворк**: PyTorch Lightning

## Технический стек

**Управление зависимостями**: uv

**Обучение**: PyTorch Lightning

**Логирование**: MLflow + TensorBoard (train_loss, val_loss, train_f1, val_f1,
test_f1, train_acc, val_acc, test_acc)

**Управление данными**: DVC (S3/Yandex Object Storage)

**Конфигурация**: Hydra (иерархические yaml + CLI)

**Качество кода**: Ruff + pre-commit

## Setup

### 1. Клонирование репозитория

```bash
git clone https://github.com/piton22/weather-recognition.git
cd weather-recognition
```

### 2. Установка зависимостей

```bash
uv sync
```

### 3. Запуск всех pre-commit хуков

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### 4. Скачивание данных (автоматическое)

Данные скачиваются АВТОМАТИЧЕСКИ при первом запуске обучения! **Как это
работает:**

- В `train.py` вызывается `pull_data()` **перед** созданием DataModule
- DVC автоматически загружает `dataset/` из S3/Yandex Object Storage
- При последующих запусках проверяется актуальность данных

### Настройка MLflow (опционально)

```bash
uvx mlflow server --host 127.0.0.1 --port 8080
```

## Train

Перед обучения заходим в python-пакет

```bash
cd weather_recognition
```

### Базовое обучение

```bash
uv run train.py
```

ResNet18, 10 эпох, batch_size=512.

### Обучение с кастомными параметрами

Быстрая проверка (1 эпоха, SimpleCNN, batch_size=8) **(Рекомендуется для
проверки)** :

```bash
uv run train.py model=simplecnn data.batch_size=8 train.max_epochs=1
```

SimpleCNN:

```bash
uv run train.py model=simplecnn
```

Меньше эпох:

```bash
uv run train.py train.max_epochs=5
```

Малый батч:

```bash
uv run train.py data.batch_size=64
```

### Доступные конфигурации

#### Модели (model=)

- `resnet18` — ResNet18 (по умолчанию)
- `simplecnn` — Легковесная CNN

#### Параметры данных (data.\*)

- `data.batch_size` — Размер батча (по умолчанию: 512)
- `data.num_workers` — Количество воркеров (по умолчанию: 4)

#### Параметры обучения (train.\*)

- `train.max_epochs` — Количество эпох (по умолчанию: 10)

#### Параметры модели (model.\*)

- `model.lr` — Learning rate (по умолчанию: 1e-3)
- `model.weight_decay` — Weight decay для Adam (по умолчанию: 1e-4)
- `model.scheduler.patience` — Patience для ReduceLROnPlateau (по умолчанию: 3)
- `model.scheduler.factor` — Фактор уменьшения LR (по умолчанию: 0.5)

#### Параметры логирования (logging.\*)

- `logging.save_dir` — Директория для TensorBoard логов (по умолчанию: logs)

- `logging.name` — Имя эксперимента TensorBoard (по умолчанию:
  weather_recognition)

## Результаты обучения

- **Модель:** `models/<model_name>/final_model.pt`
- **Метрики:** MLflow/TensorBoard (train_loss, val_loss, train_acc, val_acc,
  test_acc)
- **Логи Hydra:** `outputs/<date>/<time>/`
- **Lightning checkpoints:** `logs/weather_recognition/version_X/checkpoints/`

## Просмотр результатов в MLflow

[http://127.0.0.1:8080](http://127.0.0.1:8080)

**Содержимое логов MLflow**:

- train_loss/val_loss
- метрики accuracy
- гиперпараметры
- git commit ID.

## Структура проекта

```
weather-recognition/
├── weather_recognition/             # Основной пакет проекта
│ ├── data/                          # Модули для работы с данными
│ │ ├── datamodule.py                # PyTorch Lightning DataModule
│ │ ├── dataset.py                   # WeatherDataset (PyTorch Dataset)
│ │ ├── dvc_utils.py                 # Утилиты для DVC (pull_data)
│ │ └── transforms.py                # Трансформации и аугментации
│ ├── models/                        # Архитектуры моделей
│ │ ├── model_baseline.py            # SimpleCNN
│ │ └── model_resnet.py              # ResNet18Classifier
│ ├── litmodule.py                   # PyTorch Lightning Module
│ └── train.py                       # Training pipeline
├── conf/                            # Hydra конфигурации
│ ├── config.yaml                    # Главный конфиг
│ ├── data/                          # Конфиги данных
│ │ └── datamodule.yaml              # DataModule параметры
│ ├── model/                         # Конфиги моделей
│ │ ├── resnet18.yaml                # ResNet18
│ │ └── simplecnn.yaml               # SimpleCNN
│ ├── train/                         # Конфиги обучения
│ │ └── trainer.yaml                 # Trainer параметры
│ ├── mlflow/                        # Конфиги MLflow
│ │ └── mlflow.yaml                  # MLflow tracking
│ └── logging/                       # Конфиги логирования
│   └── tensorboard.yaml             # TensorBoard
├── train.py                         # Hydra CLI entrypoint
├── dataset.dvc                      # DVC-трекер данных
├── .dvc/                            # Внутренние файлы DVC
├── .dvcignore                       # Файлы, игнорируемые DVC
├── pyproject.toml                   # Конфигурация проекта и зависимости (uv)
├── uv.lock                          # Lock-файл зависимостей
├── .pre-commit-config.yaml          # Конфигурация pre-commit хуков
├── .gitignore                       # Игнорируемые Git файлы
├── LICENSE                          # Лицензия проекта
└── README.md                        # Документация проекта

```

## Воспроизводимость

Для полной воспроизводимости экспериментов в этом проекте используются несколько
уровней фиксации окружения и настроек:

**Версия кода:** В каждом запуске обучения в MLflow логируется текущий Git
commit ID, что позволяет однозначно восстановить состояние репозитория на момент
эксперимента.

**Зависимости:** Все версии библиотек зафиксированы в uv.lock, поэтому окружение
можно детерминированно восстановить через uv sync.

**Данные:** Датасет версифицирован с помощью DVC и подтягивается через dvc pull,
что гарантирует одинаковые данные для всех запусков.

**Конфигурация:** Все параметры обучения, модели и данных описаны в
Hydra-конфигах (conf/) и сохраняются в outputs/<date>/<time>/, вместе с
финальной использованной конфигурацией.

**Random seed:** Фиксированный seed (по умолчанию 42) задаётся в конфиге и
применяется через PyTorch Lightning, что повышает детерминизм обучения (в том
числе для data split).

## Лицензия

См. файл [LICENSE](LICENSE)
