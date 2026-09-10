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
