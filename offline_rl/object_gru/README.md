# Object Transformer + GRU

Альтернативная policy из предложенного описания. Обучение — **behavior cloning
(imitation learning)**, не IQL: сеть предсказывает действия игрока из JSON.
StARformer и существующий IQL остаются доступны в прежних файлах.
Новые зависимости и повторное извлечение trajectories не нужны.

## Файлы

- `features.py` — выбор доступных признаков из общего кодировщика наблюдений.
- `model.py` — Transformer объектов, GRU и три выходные головы.
- `train.py` — настройки и loss новой модели; общий training engine отвечает
  за разбиение по боям, sampling, метрики, checkpoint и resume.
- `predict_action.py` — предсказание по JSON/JSONL, история и маски допустимости.
- `../../predict_video_object_gru.py` — распознавание видео и отрисовка рекомендаций
  через существующий `predict_video_actions.py` / `predict_video_kalman.py`.

Отдельная копия dataset не нужна: используется `offline_rl/dataset.py`.
JSON читаются без изменений. Vocabulary, нормализация и параметры истории
сохраняются в checkpoint и повторно используются при предсказании.

## Архитектура и адаптация к вашим данным

По умолчанию история — 8 состояний, максимум — 64 объекта на состояние.
При 5 Гц это последние 1.4 секунды между первым и последним состоянием.
Если объектов больше лимита, возникает ошибка: молча отбрасывать их нельзя.
Лимит можно увеличить через `--max-units` при новом обучении.

Для каждого объекта используются embedding класса размером 32 и 6 чисел:
сторона (`ally=+1`, `enemy=-1`, `unknown=0`), нормализованные x/y центра клетки,
уверенность классификации, уверенность детектора, флаг предсказания Калмана.
Проекция даёт 128 признаков. Transformer: 3 слоя, 4 attention heads,
feed-forward 512. CLS собирает представление арены. Порядок списка и track_id
не кодируются; маски исключают padding. Пустая арена допустима.

Дополнительные ветви:

- Четыре башни `ally_1`, `ally_2`, `enemy_1`, `enemy_2`: HP, наличие измерения,
  свежесть, уверенность и возраст HP → 64 признака. Неизвестный HP не означает
  разрушенную башню. Королевские башни не добавляются.
- Четыре карты слева направо: embedding 32, confidence, known/nonempty masks →
  64 на карту, всего 256 признаков. Позиция карты в руке сохраняется.
- Свой эликсир с availability mask, прошедшее время видео и интервал от
  предыдущего наблюдения → 32 признака. Время видео не выдается за таймер боя.

Конкатенация 480 признаков → проекция 512 → однонаправленная GRU
(2 слоя, hidden size 256). Головы: WAIT/PLAY (2), слот карты (4), клетка (576).
Поле сохраняет исходные **32 строки × 18 столбцов**, а не 16 × 18 из примера.
Индекс клетки внутри модели: `(row - 1) * 18 + (column - 1)`.
Наружу возвращаются исходные `slot`, `row`, `column`, начиная с 1.
Голова позиции — общая для слотов, как в предложенном варианте; это единое
распределение по клеткам, не две независимые головы строки и столбца.

Нет выдуманных unit HP, скоростей, возраста юнита, следующей карты, стоимости,
эликсира соперника, remaining time, фазы или счёта. История действий, reward,
return-to-go, результат боя и будущие наблюдения не входят в модель.

GRU каждый раз обрабатывает ограниченное окно с нулевым начальным hidden state;
это одинаково при обучении и предсказании. В начале боя выбирается последний
реальный шаг, не последняя padding-позиция. `reset()` начинает новую историю.

GRU на GPU использует native PyTorch CUDA-операции: cuDNN отключается только
на время её forward. В текущем Windows-окружении минимальный тест cuDNN GRU
с backward воспроизвёл аварийное завершение процесса с `0xC0000409`, тогда как
native-вариант завершился нормально. Это не переключение на CPU и не глобальная
настройка cuDNN для детекторов или других моделей.

## Обучение — PowerShell

```powershell
Set-Location C:\Users\aantr\cr_bot\v2

.\venv\Scripts\python.exe .\offline_rl\object_gru\train.py .\offline_rl\trajectories `
  --output .\runs\offline_rl\object_gru_first `
  --epochs 50 `
  --batch-size 16 `
  --sequence-length 8 `
  --max-units 64 `
  --sampling balanced `
  --play-weight 1 `
  --validation-fraction 0.2 `
  --dropout 0.1 `
  --weight-decay 0.01 `
  --patience 0 `
  --device 0 `
  --num-threads 4
```

После PowerShell-символа продолжения строки (обратной кавычки) не должно быть
пробелов. Output должен быть новым/пустым каталогом. При нехватке GPU-памяти
начните новый запуск с `--batch-size 8` или `4` (для balanced — чётное число).

Loss: `CE(type) + 0.5 * CE(slot|PLAY) + 0.5 * CE(cell|PLAY, known position)`.
Метки соседних WAIT-кадров не превращаются в PLAY. Unknown-действия не становятся
WAIT. У частично распознанного PLAY без достоверной клетки обучаются только
тип и карта. По умолчанию каждый batch содержит 50% достоверных PLAY и 50% WAIT;
дополнительный большой `play-weight` обычно не нужен.

Validation содержит целые отдельные бои с естественной частотой действий,
без oversampling. При трёх видео это два train и одно validation (seed 42).
Столь малая выборка не гарантирует перенос на новые бои.

- `best.pt` — минимум validation loss.
- `best_play.pt` — максимум validation F1 для PLAY/WAIT при пороге 0.5;
  при одинаковом F1 выбирается меньший loss. Это не оценка побед или качества
  выбора карты/клетки: для них смотрите отдельные метрики.
- `last.pt` — последняя эпоха, вместе с optimizer/RNG для продолжения.
- `history.json` — loss, precision/recall/F1, пропущенные и ложные PLAY,
  точность карты/клетки и полного действия по эпохам.

Если в оценочном split нет обоих классов, `best_play.pt` не создаётся — будет
предупреждение. Тогда используйте `best.pt` и проверьте разбиение/метки.
Для проверки запоминания всех трёх видео можно начать другой запуск с
`--validation-fraction 0`: best-файлы будут оцениваться на естественном
train-потоке, а не на отложенных боях. Это **не** проверка обобщения.

Продолжение до 100 эпох суммарно (не ещё 100):

```powershell
.\venv\Scripts\python.exe .\offline_rl\object_gru\train.py `
  --resume .\runs\offline_rl\object_gru_first\last.pt `
  --epochs 100 --device 0 --num-threads 4
```

Старые StARformer/IQL checkpoint несовместимы с этой архитектурой. Начните
новое обучение; resume использует только собственные checkpoint и проверяет
неизменность исходных JSON. Архитектуру и sampling на resume менять нельзя.

## Предсказания на видео — PowerShell

```powershell
.\venv\Scripts\python.exe .\predict_video_object_gru.py `
  .\screenshots\my_dataset\oyassuu-hog-top-10\oyassuu-hog-top10_00.03.08.512-00.06.36.525-seg02.mp4 `
  --checkpoint .\runs\offline_rl\object_gru_first\best_play.pt `
  --device 0 `
  --state-fps 5 `
  --show-cards `
  --play-threshold 0.5
```

Для проверки обобщения укажите видео, не входившее в train. Здесь seg02 дан
как существующий пример пути, а не как гарантированно независимый test.
Частота состояний должна совпадать с извлечением trajectories; у текущих — 5 Гц.
OCR HP включён по умолчанию. Не отключайте его при предсказании, если обучались
на состояниях с HP. GUI показывает совет и `P(play)`, но не нажимает на игру.
Для сохранения добавьте `--output-video new_preview.mp4` и/или
`--predictions new_predictions.jsonl`; существующие файлы не перезаписываются.

Порог можно снизить, например, до `--play-threshold 0.35`, но это увеличивает
также ложные PLAY. Balanced training меняет распределение классов, поэтому
`P(play)` не является калиброванной вероятностью успешного розыгрыша.

## Предсказания из JSON и текущего состояния

Без повторного запуска YOLO/OCR:

```powershell
.\venv\Scripts\python.exe .\offline_rl\object_gru\predict_action.py `
  .\runs\offline_rl\object_gru_first\best_play.pt `
  --replay .\offline_rl\trajectories\oyassuu-hog-top10_00.03.08.512-00.06.36.525-seg02-e4167584e07a.json `
  --device 0 --limit 100
```

В коде:

```python
from offline_rl.object_gru.predict_action import ObjectGRUPredictor

policy = ObjectGRUPredictor("runs/offline_rl/object_gru_first/best_play.pt", device="0")
policy.reset("battle-1")
# observation — одно наблюдение в формате trajectories, без будущего действия.
policy.observe(observation)
action = policy.predict()
```

Каждое следующее измеренное состояние передавайте в `observe`, с возрастающим
`timestamp_ms`. Не добавляйте рекомендованное действие как реально выполненное.
Для потока JSONL есть `--input states.jsonl` или `--input -`: одна строка —
`{"observation": {...}}`; между боями — `{"reset": true, "battle_id": "new"}`.

Маски применяются к предсказанию: пустые/неизвестные карты заблокированы.
API принимает `allowed_slots[4]`, `allowed_cells[32,18]` или `[4,32,18]`.
Для видео `--allowed-cells mask.json` задаёт такие же per-card ограничения.
Клетки `#` сами по себе не запрещаются. Типы карт и правила постановки не
угадываются по названию. При `--card-costs costs.json` (JSON имя → стоимость)
проверяется доступный эликсир; без этого файла стоимость не проверяется, что
отражается в предупреждении и поле `constraints.elixir_checked`.

Проверка реализации из `v2`:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_offline_rl_object_gru.py" -v
```
