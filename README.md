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
