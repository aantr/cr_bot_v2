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

Unknown cards, low confidence, mismatched pre-action hands, multiple actions in
one interval and reused empty transitions are masked with `action_valid=false`.
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
Use `loss_mask` for play/noop and `play_loss_mask` for card/slot/coordinate heads.

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
