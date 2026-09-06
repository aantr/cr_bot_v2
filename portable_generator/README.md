# Переносимый генератор KataCR

Эта папка не импортирует пакет `katacr` и может быть целиком скопирована в другой
Python-проект. Нужен Python 3.10+.

## Что необходимо перенести отдельно

Генератор собирает изображения из вырезанных PNG-сегментов. Скопируйте исходный
каталог `Clash-Royale-Detection-Dataset` либо передайте путь к уже существующему.
Ожидаемая структура:

```text
Clash-Royale-Detection-Dataset/
└── images/
    └── segment/
        ├── backgrounds/
        ├── background-items/
        ├── archer/
        └── ...
```

## Установка и запуск

```powershell
python -m pip install -r portable_generator/requirements.txt
python -m portable_generator.generator `
  --dataset-root C:\path\to\Clash-Royale-Detection-Dataset `
  --output generation `
  --count 100 `
  --units-min 10 `
  --units-max 15
```

Для каждого изображения количество обычных юнитов выбирается случайно из
интервала `--units-min ... --units-max` включительно. Значения по умолчанию —
от 10 до 15.

Перед генерацией каталог `--output` очищается от старых изображений. Вместе с
ними удаляются одноимённые `.txt`-аннотации и прежний `yolo_annotations.txt`.
Другие файлы и вложенные каталоги не затрагиваются.

Рядом с каждым `gen_N.jpg` будет создана стандартная YOLO-аннотация `gen_N.txt`.
Включённый `classes.yaml` задаёт три выходных класса: `0: bar`,
`1: bar-level`, `2: elixir`. Режим `--generation-filter bars-elixir`
записывает в итоговые label-файлы только эти классы. Для другого набора
передайте `--classes-yaml path\to\data.yaml`.

## Использование из кода

```python
from portable_generator import Generator

generator = Generator(
    dataset_root=r"C:\path\to\Clash-Royale-Detection-Dataset",
    seed=42,
    map_update={"mode": "dynamic", "size": 5},
    generation_filter="bars-elixir",
)
generator.add_tower()
generator.add_unit(n=12)
image, boxes, pil_image = generator.build(
    save_path="generation/example.jpg",
    generate_ann=True,
)
```

Основной публичный код находится в `generator.py`; конфигурационные таблицы
вынесены в `generation_config.py`, чтобы генератор оставался читаемым.
