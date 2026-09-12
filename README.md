## Object Transformer + GRU policy

An alternative behavior-cloning policy now lives in `offline_rl/object_gru/`:
per-frame object Transformer (3 layers / 4 heads) + 8-state GRU (2 layers,
hidden 256) + WAIT/PLAY, four card slots and 576 deployment cells. It consumes
the **same trajectory JSONs**, without re-extraction or fabricated missing data.
StARformer and IQL remain available through their existing entry points.

See [architecture, feature adaptation and full PowerShell commands](offline_rl/object_gru/README.md).
Train with `offline_rl/object_gru/train.py`; preview video with
`predict_video_object_gru.py`. These entry points use their own checkpoints.

## Unit team from level-bar color

`predict_video_kalman.py` counts tinted pixels in the clean detected level-bar
crop (OpenCV BGR), rather than comparing weighted mean channels:

- Blue: `B > max(G, R)` and `B - max(G, R) >= FIELD_COLOR_BLUE_MIN_DOMINANCE`.
- Red: `R > max(G, B)` and `R - max(G, B) >= FIELD_COLOR_RED_MIN_DOMINANCE`.
- Both constants default to **10** on the 0..255 channel scale. Lower values
  admit weaker tints; even 0 excludes equal channels, gray, white and black.
- More blue pixels means `ally`; more red means `enemy`. Equal counts, including
  zero evidence, mean `unknown` (gray field cell). Each pixel has one vote.

The same result drives field coloring and exported unit `side`, including the
shared video advisor and trajectory extraction. Kalman-only tracks keep the last
measured pixel counts. `mean_color_bgr` remains diagnostic only;
`objects[].level_bar_pixel_counts` stores `[blue_count, red_count]`.
Existing trajectory JSONs are unchanged: re-extract videos to update their sides;
`--relabel-json` alone does not rerun pixel recognition.

## Offline RL trajectories

Run from `v2` with its virtual environment. Each video must contain one complete
battle at normal playback speed; trim menus or unrelated footage beforehand.
The result is for the player whose four cards are visible at the bottom.

```powershell
python build_trajectories.py battle.mp4 --result win
python build_trajectories.py battle.mp4 --result loss --output-dir my_trajectories
```

The implementation is in `offline_rl/build_trajectories.py`; the `v2` entry
point forwards to it. `--result` accepts `win`, `loss` and `draw`.
Models, crops and recognition thresholds come from the shared
`predict_video_kalman.py` pipeline and `model_paths.py`. No preview or annotated
video is produced. OCR runs synchronously at its configured video-time deadlines.

Default output: `offline_rl/trajectories/<video-stem>-<sha256-prefix>.json`.
An existing trajectory is only replaced with `--overwrite`. Each JSON contains:

- `metadata`: source SHA256, resolution, models, vocabularies, crops, thresholds,
  field layout and processing time;
- `observations`: timestamp/frame index, classified units and their confidence,
  side, 1-based row/column, Kalman flags, hand slots, elixir and confirmed tower HP;
- `field_cells` within each observation: sparse unit counts by cell, side and
  class, preserving multiple units in a cell. Elixir drops are excluded;
- `transitions`: adjacent `state_index`/`next_state_index`, actual `dt_ms`,
  action, `action_valid`, invalid reasons, reward components, terminal flag,
  return-to-go and discounted return-to-go;
- `events`, `reward_evidence`, `uncertainty_intervals` and `metrics`.

OpenCV may decode slightly fewer frames than the container reports. The default
EOF tolerance requires the shortfall to be **both** at most 1 second and at most
2% of the reported frame count. Accepted differences appear in
`metadata.video_eof` and `metrics.warnings`. No frames are fabricated; the supplied
battle result is applied to the last processed state. A larger shortfall still
fails rather than treating a substantially interrupted recording as complete.
Adjust `--eof-tolerance-seconds` / `--eof-tolerance-fraction` when needed;
`--eof-tolerance-seconds 0` requires the reported count to be reached.

Recognition defaults to 30 Hz (`--detection-fps`); states default to 5 Hz
(`--state-fps`). The last processed frame is retained. A play is assigned to
the state strictly before its inferred empty-slot time, not the frame that
confirms the elixir event 300 ms later. Future card/HP classifications are never
backfilled into earlier states. Coordinates and hand slots are 1-based;
row 1 is at the top, column 1 at the left; `#` cells are accepted.

Unknown cards, low confidence in the played slot, mismatched pre-action hands,
genuinely multiple actions and conflicting empty-transition matches are masked
with `action_valid=false`. Exact repeat confirmations are consolidated; uncertain
coordinates use a separate position mask (see action-label repair below).
Unmatched empty slots and apparent elixir spends also mask nearby intervals.
Use this mask for action losses; do not delete those intervals and collapse
elapsed time. No-op labels are inferred and can still contain missed detections.

Default reward per transition:

```text
r = (enemy_damage - ally_damage) / 1000
    + (enemy_towers_destroyed - ally_towers_destroyed)
    + terminal_reward

terminal_reward = +5 for win, -5 for loss, 0 for draw (last transition only)
```

Only confirmed OCR readings of the four side towers contribute damage.
The first confirmed HP is a baseline; missing/stale text is never zero.
Each decrease below a tower's previously accepted minimum is rewarded once,
preventing repeated rewards from high/low OCR flicker. Increases are ignored.
A single drop greater than 2500 HP is rejected as suspect. Destruction requires
a confirmed zero. These conservative rules can miss real healing or large hits;
there is no king-tower/destruction-image recognizer in this version.
The terminal outcome still works if HP is unavailable.

Weights/filters are configurable via `--damage-scale`, `--tower-reward`,
`--outcome-reward`, `--max-hp-drop` and `--min-card-confidence`.
`--no-tower-hp` explicitly disables OCR and produces terminal-only rewards.
Discounting uses `gamma_per_second ** (dt_ms / 1000)`, default
`--gamma-per-second 0.99`; terminal transitions have zero bootstrap discount.
Ordinary return-to-go is the undiscounted sum of future rewards.

Metrics include resolved events, valid play/no-op counts, valid action fraction,
invalid reasons, unmatched hand transitions, HP coverage per tower, accepted
damage, total reward, unknown unit fraction and Kalman prediction fraction.
These measure dataset quality, not detector accuracy or policy win rate.
The supplied result is a label, not an OCR prediction. Game clock/phase are
currently unknown, and timestamp is elapsed video time.

Split training/validation by `source_sha256` (whole battles), never by frames.
For policy evaluation later, use held-out action accuracy/card accuracy,
coordinate error on valid plays, and actual win rate in separately run games.

## Action-label repair

New builds use `action_label_version=2`. A confirmed play checks the played
slot's confidence, not all four hand slots; uncertain unrelated cards no longer
discard a reliable action. Noop labels still require reliable hand evidence.

Events with exactly the same card, slot, empty frame and action timestamp are
linked as duplicates. All raw candidates remain in `events` for audit. Matching
cells keep full supervision; conflicting cells set `action.position_valid=false`
and train only play/card/slot, never a guessed/averaged location. Conflicting
cards/times or genuinely multiple plays still invalidate the action. Late event
timestamps are corrected only when saved full-rate hand evidence gives a unique
earlier empty onset and a reliable matching pre-action state. Missing evidence
does not justify changing a card or slot.

Rebuild labels from an existing JSON without rerunning detectors (from `v2`):

```powershell
python offline_rl/build_trajectories.py --relabel-json offline_rl/trajectories/last_20_percent-fe680f8c497c.json --output-dir offline_rl/trajectories_repaired
```

This mode uses the saved configuration/result; original observations and rewards
are preserved. It refuses to overwrite the source JSON, even with `--overwrite`.
Legacy files lack full-rate hand evidence: unmatched intervals are preserved,
but missing card/slot evidence cannot be reconstructed. New files also retain
`empty_transitions` and `elixir_drops` for later relabeling.

For the existing `last_20_percent` JSON, relabeling gives 8 usable plays instead
of 3: 5 with known cells and 3 with ambiguous cells. Three duplicate detections
are merged; three other candidate actions remain invalid. No new detections or
card classes were invented. The repaired copy is in `offline_rl/trajectories_repaired`.

Train a NEW run on the repaired directory (do not mix original and repaired
copies of the same battle, and do not resume the old run on changed labels):

```powershell
python offline_rl/train.py offline_rl/trajectories_repaired --validation-fraction 0 --epochs 30 --output runs/offline_rl/repaired
```

This remains a one-battle smoke experiment, not proof of generalization. More
reliable play examples and matching online/training history are still needed.

## PyTorch trajectory dataset

`offline_rl/dataset.py` loads the trajectory JSONs and creates causal windows
without running recognition. From `v2`, inspect a real batch with:

```powershell
python offline_rl/dataset.py offline_rl/trajectories --sequence-length 32 --batch-size 4
```

Use directly from Python (works with one battle):

```python
from torch.utils.data import DataLoader
from offline_rl.dataset import TrajectoryDataset

dataset = TrajectoryDataset("offline_rl/trajectories", sequence_length=32)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
batch = next(iter(loader))
encoding = dataset.encoding_config()  # Store alongside the model checkpoint.
```

There is one window ending at every transition; early windows are padded on the
right. Windows never cross battles. By default `supervise="last"` trains only on
the final real step of each window, so overlapping context is not supervised
repeatedly. `supervise="all"` enables losses at all valid real steps instead.
`stride` can subsample ending steps; the final transition is always included.
Masked windows remain in the dataset; skip an action-loss update when its batch
has no true `loss_mask` entries. Mean cross entropy over only ignored targets
would otherwise produce NaN.

`states` contains normalized grid counts (ally/enemy/unknown channels), individual
unit class IDs, sides, zero-based cell coordinates, classification/detection
confidence, Kalman flags and a unit mask. Overlapping units are preserved.
Arbitrary track IDs are not model features. `max_units=128` sets the padded unit
capacity; exceeding it raises an error asking to increase capacity.
`terrain_mask` marks `#` cells but does not forbid placing objects there.

Other state tensors contain four hand IDs/confidences, elixir, four side-tower
HPs, HP known/fresh masks and observation age, elapsed time, previous elapsed
step duration and optional remaining game time/phase. Missing HP has a false
known mask; confirmed zero has a true known mask. A retained stale value has a
false fresh mask. `hand_nonempty_mask` is not an elixir-affordability/legality mask.
Normalization uses fixed configurable scales from `Normalization`, shared
between training, validation and prediction; it never divides time by a battle's
eventual length or fits statistics on validation data.

`previous_actions` and `previous_rewards` are shifted across the full battle
before slicing windows. Episode start uses BOS. A masked or out-of-vocabulary
previous action becomes UNK with all card/coordinate fields cleared; its observed
reward remains available independently. Targets contain `action_type` (0=noop,
1=play), `card_id`, `card_slot` (0..3), `row` (0..31) and `column` (0..17).
Unavailable or unsupervised targets are -100 for CrossEntropyLoss's ignore_index.
Use `loss_mask` for play/noop, `play_loss_mask` for card/slot, and
`position_loss_mask` for row/column. A known play with ambiguous location retains
card supervision but has row/column targets -100 and zero previous-coordinate
tokens; zero means unknown here, not a real field cell.

`attention_mask=True` means a real timestep. `padding_mask=True` and
`causal_mask=True` mean blocked attention. After DataLoader collation, use
`batch["causal_mask"][0]` as the shared [T,T] PyTorch attention mask and
`batch["padding_mask"]` as the [B,T] key-padding mask. Real states are followed by
padding, and causal attention prevents later states leaking into earlier queries.
Rewards, terminal flags and return-to-go are returned separately from states;
feed return-to-go only when explicitly training a return-conditioned policy.

With at least two distinct battles, split before fitting the shared vocabulary:

```python
from offline_rl.dataset import create_train_val_datasets

train_dataset, val_dataset = create_train_val_datasets(
    "offline_rl/trajectories", validation_fraction=0.2, seed=42, sequence_length=32,
)
```

Copies with the same source SHA256 stay on the same side of the split. Vocabulary
IDs come from training model class names and training observations; validation
uses those exact IDs and maps unseen classes to UNK, masking unencodable plays.
For one battle, use `TrajectoryDataset` directly or `validation_fraction=0`,
which returns `val_dataset=None`. Save `encoding_config()` with checkpoints;
restore vocabularies via `Vocabulary.from_dict(...)` and normalization via
`Normalization(**...)` for subsequent datasets and inference.

## StARformer imitation policy

`offline_rl/starformer.py` implements a project-specific StARformer-inspired
local-then-temporal policy, not a reproduction of the original architecture or
a loader for its pretrained weights. `train.py` learns imitation without
optimizing returns or consuming return-to-go. The separate `train_iql.py` trains
the same actor with offline RL (see below). `predict_action.py` loads either
checkpoint kind and maintains observed battle history for recommendations.
Neither pipeline executes actions in the game.

Example from `v2`, using the `dataset` and `batch` created above:

```python
from offline_rl.starformer import StARformer, StARformerConfig

config = StARformerConfig.from_encoding_config(dataset.encoding_config())
model = StARformer(config)  # Random weights until trained or restored.
outputs = model(batch)
model.eval()
recommendation = model.predict(batch)  # Last REAL step, not the padded last index.
```

Move all input tensors recursively to the same device as the model for GPU use.
Defaults: width 128, 4 attention heads, 1 local layer, 3 temporal layers,
dropout 0.1, and context length from the dataset. Local attention combines 48
field patch tokens, masked individual units, four hand slots, tower HP, elixir,
time, previous executed action and previous observed reward. Temporal attention
is causal; the model constructs its own mask and requires nonempty right-padded
sequences. Targets, current rewards, future returns and terminal flags are not
inputs. Unit ordering carries no positional meaning; their cells do.

`forward()` returns raw logits, with zero logits at padded timesteps:

- `action_type`: `[B,T,2]`, 0=noop, 1=play.
- `card_slot`: `[B,T,4]`, zero-based hand position (not a vocabulary class).
- `row_by_slot`: `[B,T,4,32]` and `column_by_slot`: `[B,T,4,18]`.

Coordinate heads are conditioned on each hand slot. In the training loop,
apply action-type cross entropy at `loss_mask`, slot cross entropy at
`play_loss_mask`, and coordinate cross entropy at `position_loss_mask`.
Select coordinate logits using the ground-truth slot:

```python
play = batch["position_loss_mask"]
if play.any():
    slots = batch["targets"]["card_slot"][play]
    rows = outputs["row_by_slot"][play].gather(
        1, slots[:, None, None].expand(-1, 1, 32)
    ).squeeze(1)
    # cross_entropy(rows, batch["targets"]["row"][play]); similarly for columns.
```

Skip loss components with no valid targets. `card_id` is obtained from the
selected hand slot, so there is no separate card-class loss. Save
`model.state_dict()`, `config.to_dict()` and `dataset.encoding_config()` together
in every checkpoint; reconstruct with `StARformerConfig(**saved_config)`.

`model.eval(); model.predict(batch)` selects play/noop, then the slot, then the
highest-scoring permitted cell for that slot. It returns one tensor per field
(`action_type`, `card_slot`, `card_id`, `row`, `column`) of shape `[B]`.
Slots and cells are zero-based, and all non-type fields are -1 for noop.
The decoder excludes empty/unknown cards and falls back to noop when no play is
allowed. Optional boolean `allowed_slots[B,4]` and
`allowed_cells[B,4,32,18]` masks support card costs and placement restrictions;
the caller must compute these rules. Without those masks, known nonempty slots
and every field cell are eligible. `#` cells are not automatically forbidden.
This helper does not manage live history or execute actions in the game.

Run dataset and model tests from the project root:

```powershell
.\v2\venv\Scripts\python.exe -m unittest discover -s v2/tests -p "test_offline_rl_*.py" -v
```

## IQL: обучение offline RL

Новый путь обучения не заменяет imitation: `offline_rl/train.py` остаётся
behavior cloning, а `offline_rl/train_iql.py` обучает **Implicit Q-Learning**.
Дополнительные библиотеки или симулятор не нужны: используется установленный PyTorch.
Формулы сверены с [реализацией авторов IQL](https://github.com/ikostrikov/implicit_q_learning).
Здесь они адаптированы к дискретному действию `wait` или `(slot, row, column)`.

Новые файлы:

- `offline_rl/iql_dataset.py`: пары историй до действия и после него, действие,
  награда, уже рассчитанный временной discount. Загружает существующие JSON боёв.
- `offline_rl/iql.py`: actor, два Q-критика, V-сеть, две замороженные целевые
  Q-сети с EMA. У каждой сети независимый StARformer-энкодер истории.
- `offline_rl/train_iql.py`: обучение, статистика, validation по отдельным боям,
  сохранение всех сетей/оптимизаторов и восстановление из `last.pt`.
- `offline_rl/history_features.py`: общий контракт входов обучения и предсказания.

На каждом шаге обучения:

```text
q_data = min(target_Q1(history, action), target_Q2(history, action))
V_loss = mean(expectile_loss(q_data - V(history), tau=0.7))
target = reward / normalization.reward + discount * V(next_history)
Q_loss = MSE(Q1, target) + MSE(Q2, target)
weight = min(exp(beta * (q_data - V(history))), 100), beta=3
actor_loss = mean(-stop_gradient(weight) * log_probability(dataset_action))
target_Q <- 0.995 * target_Q + 0.005 * Q
```

Bellman-target и веса actor отсоединены от градиентов; actor не изменяет Q/V.
Для play логарифм вероятности складывается из type, slot и координат,
условных на выбранный слот. Для wait учитывается только type. Карта определяется
содержимым слота руки. `discount` из JSON уже учитывает время и равен нулю при
`terminated`: повторного умножения на gamma нет. `truncated` сам по себе не
обнуляет bootstrap. RTG и будущие наблюдения не подаются actor на вход.

IQL сохраняет `history_mode=observations_only`: actor и критики видят последние
наблюдения поля, руки, HP, элексира и времени, но не прошлые action/reward-токены.
Это предотвращает различие между ретроспективными метками JSON и задержанным
подтверждением действий в видео. Награды по-прежнему обучают критики через
Bellman-target. Существующие imitation-checkpoint сохраняют прежний режим входов.

### Запуск IQL

Все команды ниже выполняются **из `v2`**, интерпретатор берётся из `v2/venv`.
Сначала извлеките целые бои с правильным результатом, включая победы и поражения:

```powershell
.\venv\Scripts\python.exe build_trajectories.py battle_win.mp4 --result win
.\venv\Scripts\python.exe build_trajectories.py battle_loss.mp4 --result loss
.\venv\Scripts\python.exe offline_rl/train_iql.py offline_rl/trajectories --output runs/offline_rl/iql_first --epochs 50 --device 0 --validation-fraction 0.2
```

Для пробного запуска на имеющемся единственном исправленном бою:

```powershell
.\venv\Scripts\python.exe offline_rl/train_iql.py offline_rl/trajectories_repaired --output runs/offline_rl/iql_single --epochs 20 --device 0 --validation-fraction 0
```

Validation делится по идентичности целых боёв, до построения окон. Нельзя класть
оригинальную и исправленную копии одного боя в одну обучающую выборку: IQL
отклоняет дубликаты. При одном бое нужен `--validation-fraction 0`; это проверка
работоспособности, не оценка обобщения. По умолчанию batch=2, context=32,
max_units=128, d_model=128, dropout=0. При нехватке GPU-памяти начните новый запуск
с `--batch-size 1`; уменьшение `--sequence-length 16` также сокращает контекст.
`--device cpu --num-threads 4` позволяет обучаться без CUDA.

Продолжить обучение (epochs — суммарное число эпох, а не число добавляемых):

```powershell
.\venv\Scripts\python.exe offline_rl/train_iql.py --resume runs/offline_rl/iql_first/last.pt --epochs 100 --device 0
```

Восстанавливаются actor, Q1/Q2/V, target-Q, четыре оптимизатора, конфигурация,
разбиение файлов и RNG. Изменённые JSON, конфигурация или откат на старый checkpoint
в уже продолженном каталоге отклоняются. Для новых данных создайте новый запуск.
`--init-actor runs/offline_rl/first/best.pt` позволяет начать IQL с imitation-actor
(или другого IQL-actor), но не переносит критики/оптимизаторы. Архитектура
наследуется, словарь и нормализация должны совпасть; при переключении с imitation
изменяется контракт истории, поэтому потребуется дообучение. Можно обучать с нуля.

`history.json` содержит Q/V loss, weighted actor loss, **невзвешенный** joint
policy NLL, advantage/веса, долю ограниченных весов, play precision/recall и
точность целого действия. `best.pt` выбирается по минимальному validation NLL
(без validation — training NLL); это лишь прокси соответствия записанным
действиям, **не win rate и не оценка ценности новой политики**. `last.pt` сохраняется
после каждой эпохи. Early stopping по NLL опционален: `--patience 10`; по умолчанию
отключён, поскольку NLL не измеряет RL-качество.

### Предсказание IQL

```powershell
.\venv\Scripts\python.exe predict_video_actions.py battle.mp4 --checkpoint runs/offline_rl/iql_first/best.pt --device 0 --show-cards
.\venv\Scripts\python.exe offline_rl/predict_action.py runs/offline_rl/iql_first/best.pt --replay offline_rl/trajectories_repaired/last_20_percent-fe680f8c497c.json --limit 10
```

Оба существующих предиктора распознают тип checkpoint автоматически. Во время
предсказания запускается **только actor**; Q/V не нужны. В API достаточно
`predictor.observe(observation)` и `predictor.predict()`; между боями вызывайте
`predictor.reset()`. Используйте ту же частоту наблюдений (обычно 5 FPS) и время
от начала боя, что при извлечении. В видео-примере IQL не ждёт подтверждения
событий и не использует `--feedback-delay-ms`. Вывод — рекомендация, не обнаружение
фактического розыгрыша и не автоматическое нажатие в игре. Проверка доступности
элексира требует `--card-costs`, ограничения размещения — `--allowed-cells`.

### Ограничения разметки для IQL

Для Q нужен полностью известный action. Поэтому неизвестные/множественные
действия, неизвестные карты и `position_valid=false` исключаются из IQL. Они
не становятся wait; промежуточные наблюдения остаются в истории; Bellman-переходы
не перескакивают через пропущенные интервалы. Последнее наблюдение боя доступно
для next_history. Частичные розыгрыши по-прежнему можно использовать в imitation.

Статистика перед обучением показывает сохранённые play/noop, причины исключений,
потерянные конечные переходы и ненулевые награды. Если неизвестен action на
конечном переходе, его награда за победу/поражение **не переносится на придуманное
действие**. Большие пробелы разметки ухудшают обучение V и распространение наград:
последующее состояние может не иметь собственной надёжной обучающей пары.
Нужны качественная разметка, целые бои и разнообразие результатов.

На текущем исправленном `last_20_percent` из 251 перехода пригодны 63:
58 wait и 5 play с известной клеткой. Ещё 3 play имеют неоднозначную клетку,
185 переходов невалидны. Из 7 ненулевых наград сохраняются 3 награды за изменение
HP; 4 исключаются, включая конечную награду за победу. На этих данных можно
проверить запуск, но нельзя ожидать освоения игры или достоверного роста win rate.

Проверки (из корня проекта):

```powershell
.\v2\venv\Scripts\python.exe -m unittest discover -s v2/tests -p "test_offline_rl_*.py" -v
.\v2\venv\Scripts\python.exe -m unittest discover -s v2/tests -p "test_predict_video_actions.py" -v
```

## Train the imitation policy

### Balanced imitation batches and `best_play.pt`

To counter rare play labels, `offline_rl/train.py` now accepts
`--sampling balanced`. The old/default `--sampling natural` is unchanged.
Balanced mode requires `--supervise last` and an even batch size >=2. With
`--batch-size 4`, **every training batch contains two play and two noop windows**.
Unknown final labels are excluded from sampling, not converted to wait. Their
observations still appear in causal history. Partial-position plays participate
in type/slot learning; their coordinate targets stay masked.

The sampler (`offline_rl/balanced_sampling.py`) shuffles each class pool, covers
the larger pool once per epoch (padding the last batch if needed), and cycles
the smaller pool. This increases the number of updates per epoch and repeatedly
uses rare plays; it does not create new independent examples. Start with
`--play-weight 1` instead of combining oversampling with weights of 50 or 100.

From `v2`, train on all three current JSONs as a memorization diagnostic:

```powershell
.\venv\Scripts\python.exe offline_rl/train.py offline_rl/trajectories --output runs/offline_rl/imitation_balanced --sampling balanced --batch-size 4 --play-weight 1 --epochs 50 --validation-fraction 0 --patience 0 --device 0
```

To first check one battle, replace the source directory with one JSON path and
choose another output directory. `--dropout 0 --weight-decay 0` can be used for
that controlled overfitting check; success on it does not establish generalization.
For a held-out battle, start a new run with `--validation-fraction 0.2` instead:
with the current three identities this uses two for training and one for validation.
**Validation is never balanced.** Without validation, an extra `train_eval` pass
runs after each epoch: eval mode, no dropout, original windows/frequencies,
without repeating plays. It is still training-set evaluation, not a test set.

Saved files:

- `last.pt`: all training state for resume, including sampler RNG and best scores.
- `best.pt`: lowest validation loss; in balanced runs without validation, lowest
  natural `train_eval` loss (not the oversampled optimization loss).
- `best_play.pt`: highest play F1 on validation or natural `train_eval`. Lower
  loss breaks an F1 tie. F1 penalizes both missed and false play predictions.
  Selection uses raw `P(play)>0.5`, not your preview's custom threshold or masks.
  Both labeled play and noop must exist in the evaluation split; otherwise this
  checkpoint is not created and the trainer warns. An initial F1=0 checkpoint
  can exist: its filename alone is not evidence that the model learned to play.
- `history.json`: `train`, `validation`, and (without validation) `train_eval`,
  including `play_f1`, `play_precision`, `play_recall`, `missed_plays`,
  `false_positive_plays`, and separate card/cell/full-action accuracy.

F1 measures **exact labeled decision steps**, not temporally matched game events,
correct placement or win rate. False/missed recognition labels affect it.
Early stopping still uses loss, not F1; `--patience 0` disables it for this diagnostic.

Preview the new play-selected checkpoint at the evaluation threshold first:

```powershell
.\venv\Scripts\python.exe predict_video_actions.py screenshots/my_dataset/oyassuu-hog-top-10/oyassuu-hog-top10_00.03.08.512-00.06.36.525-seg02.mp4 --checkpoint runs/offline_rl/imitation_balanced/best_play.pt --device 0 --show-cards --play-threshold 0.5
```

Continue the same balanced run:

```powershell
.\venv\Scripts\python.exe offline_rl/train.py --resume runs/offline_rl/imitation_balanced/last.pt --epochs 100 --device 0
```

Old checkpoints remain loadable and resume as natural sampling. Changing
natural/balanced mode, batch size or loss weights requires a **new run**, not
`--resume`. `best_play.pt` has the same actor format and works in both predictors.
No recognition changes, JSON relabeling, or IQL sampling changes are involved.

Local single-battle check: `runs/offline_rl/imitation_balanced_one` was trained
from scratch for 10 epochs on `seg03` with balanced batches of 4, play weight 1,
dropout 0 and weight decay 0 (other model settings unchanged). Its
`best_play.pt` is epoch 7: on 366 original labeled steps it detects 14/15 plays,
misses 1, produces 4 false plays, and reaches F1=0.8485 at threshold 0.5.
These are training-battle replay metrics, not performance on new videos.

### Original natural-sampling workflow

Pass JSON trajectories, not raw videos. From `v2`, with at least two battles:

```powershell
python offline_rl/train.py offline_rl/trajectories --epochs 30 --batch-size 4 --device auto --output runs/offline_rl/first
```

The default split holds out 20% of battle identities (at least one). Copies of
the same battle stay together, and vocabularies are fitted only on training
files. With just one battle, explicitly disable validation:

```powershell
python offline_rl/train.py offline_rl/trajectories --validation-fraction 0 --epochs 10 --output runs/offline_rl/smoke
```

This second mode tests the pipeline but gives no held-out quality estimate.
`best.pt` then uses training loss and early stopping is disabled. The program
warns if train/validation contains no usable play labels, and stops with an
error if a split has no usable actions at all. Masked batches are skipped.

Training uses AdamW (learning rate 0.0003, weight decay 0.01), gradient clipping
at 1.0 and float32. It minimizes the sum of four mean cross-entropies: play/noop
on `loss_mask`, slot on `play_loss_mask`, and coordinates on
`position_loss_mask`. Coordinates use the
target slot during training. Rewards are historical input features, not a
return-maximization objective. `--play-weight 2` optionally increases the play
class weight in the play/noop loss; the default is 1 (no reweighting).

Output files:

- `best.pt`: lowest validation loss (or training loss without validation).
- `last.pt`: latest completed epoch, including optimizer, RNG/shuffle state,
  model configuration, dataset vocabulary/normalization, training options and
  original split with trajectory content hashes.
- `history.json`: per-epoch losses, action counts, skipped batches and metrics.

Losses are aggregated by target counts, not by averaging unequal batch means.
Metrics include play/noop accuracy, full-action accuracy, play precision/recall,
and slot/card/cell/full-action accuracy on true plays. Cell/full-action metrics
exclude plays whose position is unknown and report their denominator separately
(`positions`, `complete_actions`). Coordinate metrics use
the predicted slot, not the target slot. Metrics use raw greedy heads without
game legality masks; missing-denominator metrics are JSON `null`, not zero.
Check play-specific metrics: high overall accuracy can hide always choosing noop.

By default early stopping uses 10 validation epochs without improvement;
`--patience 0` disables it. `--min-delta` sets the minimum improvement needed to
reset patience, while `best.pt` always retains the actual lowest loss.

Resume in the same run directory:

```powershell
python offline_rl/train.py --resume runs/offline_rl/first/last.pt --epochs 60 --device auto
```

`--epochs 60` means 60 total epochs, not 60 more. Configuration, split,
optimizer and random states are restored; changed/missing trajectory files or
conflicting training options are rejected. A checkpoint that already reached
early stopping is not restarted. Start a new run to change the dataset or
hyperparameters. A fresh run refuses a nonempty output directory. Omitting
`--output` creates a timestamped directory under `v2/runs/offline_rl`.
Seeded runs and RNG restoration support reproducibility on the same backend;
different hardware/backends need not be bit-identical.

Use `--device cpu`, `--device cuda:0` (or `--device 0`) to choose a device.
Default `auto` selects CUDA when available. `--num-workers 0` is the Windows-safe
default; `--num-threads 2` can limit CPU threads. To reduce memory use, lower
`--batch-size`, `--sequence-length` or model width (`--d-model`, divisible by
`--n-heads`). `--max-units` must still accommodate every recognized unit.
Use `python offline_rl/train.py --help` for all settings.

## Predict actions from recognized states

`offline_rl/predict_action.py` loads `best.pt` or `last.pt` without needing the
original training files. It restores the saved model, vocabulary, normalization
and context length. It does not recognize video itself or click in the game.

From `v2`, inspect recommendations along an existing battle:

```powershell
python offline_rl/predict_action.py runs/offline_rl/first/best.pt --replay offline_rl/trajectories/last_20_percent-fe680f8c497c.json --limit 10 --device auto
```

Replay uses each recorded state and only the ACTUAL preceding action/reward.
It never feeds its own recommendations back as executed actions and skips the
terminal observation. This is not a simulated rollout or a win-rate evaluation.
Remove `--limit` to process every decision step. Output is JSONL on stdout;
`--output predictions.jsonl` creates a new file and refuses to overwrite one.

Use the Python API inside the recognition loop:

```python
from offline_rl.predict_action import ActionPredictor

predictor = ActionPredictor("runs/offline_rl/first/best.pt", device="auto")
predictor.reset("battle-001")
predictor.observe(first_observation)  # No previous transition at battle start.
recommendation = predictor.predict()

# At the next sampled state, report what ACTUALLY happened since the last state.
predictor.observe(next_observation, previous_action={"type": "noop"}, previous_reward=0.0)
recommendation = predictor.predict()
```

Observations have the same structure as trajectory `observations[]` / the
return value of `build_trajectories.make_observation`: `timestamp_ms`, `units`,
four `hand` entries, `elixir` and `tower_hp`, plus optional game time and phase.
Do not pass an image or the raw recognition callback dictionary (`elixir_bar`
there must first become observation `elixir`). Use milliseconds since battle
start, strictly increasing, and the same sampling cadence as training (normally
about 200 ms). Call `reset()` before every new battle; history is bounded by the
checkpoint context length and preserves preceding actions at the window edge.

An actual play is `{"type":"play", "card":"knight", "slot":1, "row":16,
"column":9}` with an optional actual `timestamp_ms` in the preceding interval.
Slots, rows and columns in this public API are **one-based**, matching trajectory
JSONs. The played card must match the preceding hand. Omitted previous action or
`previous_action_valid=False` encodes unknown, not noop. Omitted previous reward
is masked as unavailable, not treated as a measured zero. Supply measured rewards
with the same formula used to build trajectories when available.
`predict()` never changes history; `observe()` is the only append operation.

A recommendation contains `type` (`noop` or `play`), `slot`, `card`, `row`,
`column`, `index_base=1`, observation timestamp, history length, reason,
constraint status and confidence scores. Noop's card/slot/coordinates are null.
The timestamp is the observation time, not evidence that a play occurred.

Empty/unknown hand slots are always excluded. Optional controls:

- `--min-card-confidence 0.7` / constructor `min_card_confidence=0.7` masks
  low-confidence hand recognition.
- `--card-costs costs.json` / constructor `card_costs={"knight":3,"arrows":3}`
  enables elixir affordability checks. With costs enabled, missing cost or
  unknown elixir blocks that slot. Include every card that may be selected.
- `predict(slot_costs=[3, 4, None, 2])` supplies current dynamic costs and
  overrides the static cost table; None blocks a slot.
- `predict(allowed_slots=[True, False, True, True])` applies additional slot
  restrictions. `allowed_cells` is a boolean `[32,18]` mask or a separate
  `[4,32,18]` mask for each hand slot. True means allowed. Python mask arrays
  are indexed from zero even though returned coordinates are one-based.

No card prices or deployment rules are guessed. Without costs,
`constraints.elixir_checked` is false; without cell masks every field cell is
eligible, including `#`. The caller supplies placement restrictions, cooldowns
and any additional rules. If all plays are blocked, the result is noop.
Softmax confidence is not a calibrated chance of success: action-type confidence
is raw; slot and cell scores are normalized over allowed choices, and row/column
scores are marginals of that cell distribution. A forced noop can therefore
have low raw noop confidence.

For another process, stream one JSON object per line:

```powershell
python offline_rl/predict_action.py runs/offline_rl/first/best.pt --input states.jsonl
```

Each request is `{"observation": {...}, "previous_action": {...},
"previous_reward": 0.0}`; omit feedback for the first state. Optional keys are
`previous_action_valid`, `allowed_slots`, `allowed_cells`, `slot_costs`.
`--input -` reads stdin. Send `{"reset":true,"battle_id":"battle-002"}` between
battles (it returns a reset acknowledgment). Invalid input stops with its line
number rather than silently breaking action/state alignment.

The default field layout comes from `field.py`; replay uses the trajectory's
layout. `--field-layout layout.json` or constructor `field_layout=...` overrides
it with 32 strings of 18 `.`/`#` cells. Use the same layout as training.
Online features and offline training now share `ObservationEncoder` in
`dataset.py`; regression tests compare tensors exactly, including rolling windows.

## Video example with action recommendations

Both imitation and IQL support `--play-threshold` (default `0.5`) in
`predict_video_actions.py` and `offline_rl/predict_action.py`. For example, add
`--play-threshold 0.35` to your existing command to propose plays more often,
without retraining. The rule is strictly `P(play) > threshold`; equality waits,
preserving the original default behavior. Valid thresholds are finite numbers
in `[0,1]`. Lower thresholds may increase false plays; this is a decoding control,
not evidence of improved gameplay.

The video overlay and console show raw `P(play)` and the active threshold for
both PLAY and WAIT. JSONL includes `play_probability` and `play_threshold`.
`confidence.action_type` retains its original meaning (raw probability of the
returned action type). Scores are not calibrated win probabilities. Empty/unknown
cards, low hand confidence, supplied cost checks and cell/slot masks still block
plays even above threshold; the overlay explicitly shows when no allowed play
exists. API: `ActionPredictor(checkpoint, play_threshold=0.35)`.

`predict_video_actions.py` combines the existing `predict_video_kalman.py`
perception pipeline and the trained policy. From `v2`:

```powershell
python predict_video_actions.py battle.mp4 --checkpoint runs/offline_rl/first/best.pt
```

Without arguments it uses `screenshots/input_omydays_cutted.mp4` and
`runs/offline_rl/first/best.pt`. Detection/classification engines and crop settings
remain those of `predict_video_kalman.py` / `model_paths.py`.
The usual Tracking and Arena windows are shown. Tracking additionally displays
`AI: WAIT` or the recommended card, hand slot, row and column; a recommended
deployment cell is outlined in yellow on the video. Q exits. No clicks are sent.

The default recognition rate is 30 frames/video-second and policy state rate is
5 Hz (200 ms), as in the default trajectory builder. Use `--state-fps` and
`--detection-fps` to match the settings used for training. Keep the original
damage/reward settings too; optional `--damage-scale`, `--tower-reward` and
`--max-hp-drop` mirror the trajectory builder. HP OCR runs synchronously so
confirmed HP evidence follows the same timing as extraction.

Actual play events are often confirmed late. The adapter rebuilds the bounded
history from evidence already available, assigns a play to its original time,
and never treats a recommendation as an executed action. Recent unconfirmed
noops remain unknown for `--feedback-delay-ms 1000`. This is a conservative
waiting heuristic, not a guarantee of perfect event recognition. Confirmed HP
changes provide historical rewards; no win/loss outcome is guessed during video
playback. One extra predecessor state is retained at the rolling-window boundary.

Optional output and debug settings:

```powershell
python predict_video_actions.py battle.mp4 --checkpoint runs/offline_rl/first/best.pt --show-cards --output-video screenshots/policy_preview.mp4 --predictions policy_predictions.jsonl
```

Recording is opt-in, and existing output files are not overwritten. Additional
options: `--show-battlefield`, `--headless`, `--max-seconds 10` for a short check,
and `--no-tower-hp` to skip OCR (historical rewards then become unknown).
`--device` selects the policy device only; perception retains its own device
configuration. The preview processes video as fast as inference permits, not
necessarily at real-time playback speed.

Use `--card-costs costs.json` and optionally `--allowed-cells cells.json` for
affordability and placement restrictions as described above. Without them the
overlay explicitly shows that those checks are off. Default minimum hand-card
confidence is 0.7 (`--min-card-confidence`). Recommendations alone are not proof
that an action is legal, and this example has no automatic game controls.

The only extension to the shared video loop is an optional `annotation_callback`
called after recognition and before display/video writing. Existing callers of
`run_video_prediction()` behave unchanged when it is omitted.

## How to train the bars detector

Make sure `dataset` contains images, before running export_tensor_rt.py edit model_paths.py, then run from `v2`:

```powershell
.\train_bars\run_generator.bat
python train_bars\split_yolo_annotations.py generation\yolo_annotations.txt
python train_bars\train_p2.py
python train_bars\export_tensor_rt.py
```

Training sources and configuration are stored in `train_bars`. The shared
generator is stored in `portable_generator`. The generated dataset remains in
`generation`, and detector runs remain in `runs/detect/runs_game`.
The bars dataset contains only `0: bar` and `1: bar-level`.

To add real video frames with model-generated YOLO labels to a dataset:

```powershell
python train_bars\build_bars_dataset.py `
  --video screenshots\input_omydays.mp4 `
  --output bars_video_dataset `
  --stride 10
```

The preview uses `DETECTION_ENGINE_PATH`. Press Enter or `A` to save the clean
battlefield frame and its `.txt` label, Space or `S` to skip it, and `Q` or Esc
to stop. `yolo_annotations.txt` is updated after every saved frame.

## How to train the elixir detector

The elixir pipeline mirrors `train_bars`, but writes a separate one-class
dataset (`0: elixir`) and separate training runs:

```powershell
.\train_elixir\run_generator.bat
python train_elixir\split_yolo_annotations.py generation_elixir\yolo_annotations.txt
python train_elixir\train_p2.py
python train_elixir\export_tensor_rt.py
```

Generated data is stored in `generation_elixir`, and training results are
stored in `runs/detect/runs_elixir`. Before exporting or collecting real video
frames, check the elixir model paths in `model_paths.py`.

To collect real frames after the first elixir model has been exported:

```powershell
python train_elixir\build_elixir_dataset.py `
  --video screenshots\input_omydays.mp4 `
  --output elixir_video_dataset `
  --stride 10
```

## How to train yolo classification

Make sure `dataset_centered` contains images, then run from `v2`:

```powershell
python train_classification\train_classification.py
```

The generated split remains in `dataset_centered_yolo`, and classification runs
remain in `runs_game`.

## Fast video prediction

```powershell
python predict_video_fast.py `
  --input screenshots\input_omydays.mp4 `
  --output screenshots\output_tracked_fast.mp4
```

The fast variant uses both bars and elixir TensorRT detectors. Unit
classification is batched and cached by `track_id`; elixir detections are never
sent to the classifier. Use `--classification-refresh 0` to classify each track
only once, or `--no-display` for headless processing.

For Kalman-smoothed bar and elixir bounding boxes, run:

```powershell
python predict_video_kalman.py
```

Its output is written to `screenshots/output_tracked_calman.mp4`.

## Tower HP OCR in Kalman video prediction

Install the optional Python dependency from the project root:

```powershell
.\v2\venv\Scripts\python.exe -m pip install -r v2\requirements-ocr.txt
```

HP recognition uses PaddleOCR 3.x `TextRecognition`, without text detection or
orientation models. Tesseract is no longer used. The lightweight
`en_PP-OCRv3_mobile_rec` model is loaded once in a persistent subprocess
(`tower_hp_process.py`), and all four crops are passed in one batch over local
pipes. The worker does not import PyTorch or re-run the video script. This avoids
the Windows cuDNN DLL conflict between Paddle and PyTorch. The first run downloads the recognition weights; subsequent runs use
the cache. Set `TOWER_HP_MODEL_DIR` to an existing local inference-model directory
to work offline. See the [TextRecognition documentation](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html).

Install exactly one Paddle backend separately: `paddlepaddle` (CPU) or a compatible
`paddlepaddle-gpu` build (GPU). Do not install both; they share the same package
directory. `requirements-ocr.txt` only installs the OCR wrapper and leaves the
backend choice to you. The existing GPU build also supports `TOWER_HP_DEVICE="cpu"`.
For `"gpu:0"`, the Paddle GPU build must support your GPU architecture and driver.
The PyTorch/CUDA installation used by YOLO does not enable Paddle GPU
support by itself. `TOWER_HP_CPU_THREADS=4` limits CPU inference threads.
`TOWER_HP_ENABLE_MKLDNN=False` avoids a reproduced model initialization error with
the installed Paddle 3.0.0rc1. This setting trades CPU speed for compatibility.
Process isolation does not by itself fix unsupported GPU kernels.
The tested `paddlepaddle-gpu==3.0.0rc1` CUDA 12.3 build is not usable on this
RTX 5080: its first batch takes about 49 seconds and returns empty predictions.
Keep `TOWER_HP_DEVICE="cpu"` with that build. Use GPU only after installing and
validating a newer Paddle build compatible with Blackwell/CUDA 13.

The worker exits on normal shutdown or parent EOF; an `atexit` handler also closes
it on exceptions. Startup/request timeouts are controlled by
`TOWER_HP_STARTUP_TIMEOUT` and `TOWER_HP_REQUEST_TIMEOUT`. A timed-out worker is
terminated so its late results cannot be applied to another frame.

The four HP crops come from `TOWER_HP_1`, `TOWER_HP_2`, `TOWER_HP_ENEMY_1`,
and `TOWER_HP_ENEMY_2` in `model_paths.py`. Currently they are configured for
1206 x 2622 video only. Other resolutions require explicit crop coordinates;
the script does not guess them. Set `TOWER_HP_ENABLED=False` to run without OCR.

Settings near the top of `predict_video_kalman.py`:

- `TOWER_HP_FPS=10`: OCR rounds per **video** second, capped by the processing FPS.
  Each round reads four clean crops with one batched PaddleOCR call.
  OCR runs asynchronously: video inference does not wait for it. If the previous
  batch is still running, the scheduled sample is skipped instead of queued, so
  the achieved OCR rate may be lower than this setting. `batch_ms` measures the whole batch;
  `ocr_ms` is that time divided by the batch size, not a per-crop measurement.
- `TOWER_HP_MIN_CONFIDENCE=0.70`: minimum normalized OCR score, not a calibrated
  probability. `TOWER_HP_CONFIRM_READINGS=2`: consecutive accepted readings
  needed to confirm a new value. Failed readings reset the candidate.
- `TOWER_HP_MAX_VALUES`: per-tower upper bounds. The default 10000 is only a
  permissive safety limit; set the actual maximum for the battle and levels.
- `TOWER_HP_SCALE=1.0`: original color crops, without thresholding/inversion.
  The recognition model performs its own resize and normalization.
- `TOWER_HP_MODEL_NAME`: recognizer selection; changing it may download new weights.
- `show_tower_hp=True`: show a compact window with original crops, OCR inputs,
  raw text, confidence, and rejection reasons.

`tower_hp["ally_1"]` (also `ally_2`, `enemy_1`, `enemy_2`) stores `hp`,
`confidence`, `observed_at_ms`, `confirmed_at_ms`, `last_seen_at_ms`, and `stale`.
The same snapshot is attached to `frame_data["tower_hp"]` and shown on the output
video. HP starts as `None`; missing text never implies a destroyed tower or zero HP.
When OCR fails or a different value is not yet confirmed, the last confirmed HP
is retained with `stale=True`. Increases are not automatically rejected.

Each OCR round appends all raw results (including failures) and confirmed states
to `OUTPUT_VIDEO.with_suffix(".tower_hp.jsonl")`. Runs are separated by a `run`
record containing the source video, FPS, and crops. Confirmed changes also appear
in `battle_events.log`. All observation times refer to the source video;
`observed_at_ms` is the first reading of the confirmed change, not its later
confirmation time. Rapid intermediate HP changes may be missed at low OCR rates.

Run the helper tests without downloading OCR models or running GPU inference:

```powershell
.\v2\venv\Scripts\python.exe -m unittest discover -s v2/tests -p "test_tower_hp*.py" -v
```

Run the actual PaddleOCR diagnostic on `screenshots/IMG_1357.mp4`:

```powershell
.\v2\venv\Scripts\python.exe v2/tests/benchmark_tower_hp.py
```

This prints raw HP results and timings at 0, 0.1, 0.2, 1, 3 and 8 seconds and saves crop
previews in a new `screenshots/hp_diagnostic_*` directory without running YOLO.
