from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


class PathManager:
    """Small replacement for KataCR's dataset path manager."""

    def __init__(self, path_dataset: str | Path):
        self.path = Path(path_dataset).expanduser().resolve()

    def search(
        self,
        subset: str | None = None,
        part: int | str | None = None,
        video_name: str | None = None,
        name: str | None = None,
        regex: str = "",
        drop_regex: str | None = None,
    ) -> list[Path]:
        path = self.path
        if subset is not None:
            path /= subset
        if part is not None:
            path /= f"part{part}" if isinstance(part, int) else part
        if video_name is not None:
            path /= video_name
        if name is not None:
            path /= str(name)
        if not path.exists():
            return []

        matcher = re.compile(regex)
        drop_matcher = re.compile(drop_regex) if drop_regex else None
        return [
            candidate
            for candidate in sorted(path.rglob("*"))
            if candidate.is_file()
            and matcher.search(candidate.name)
            and not (drop_matcher and drop_matcher.search(candidate.name))
        ]


def build_label2colors(labels: Iterable[int]) -> dict[int, tuple[int, int, int]]:
    labels = sorted(set(int(label) for label in labels))
    count = max(len(labels), 1)
    return {
        label: tuple(
            round(channel * 255)
            for channel in colorsys.hsv_to_rgb(index / count, 0.75, 1.0)
        )
        for index, label in enumerate(labels)
    }


def plot_box_PIL(
    image: Image.Image,
    box: Sequence[float],
    text: str | None = None,
    format: str = "xywh",
    box_color: str | tuple[int, int, int] = "red",
    draw_center_point: bool = False,
) -> Image.Image:
    """Draw the subset of KataCR box visualization used by this generator."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    x0, y0, third, fourth = map(float, box[:4])
    if format == "voc":
        x1, y1 = third, fourth
    else:
        x1, y1 = x0 + third, y0 + fourth
    draw.rectangle((x0, y0, x1, y1), outline=box_color, width=2)
    if draw_center_point:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=box_color)
    if text:
        text_box = draw.textbbox((x0, y0), text)
        draw.rectangle(text_box, fill=box_color)
        draw.text((x0, y0), text, fill="black")
    return image


def write_yolo_annotations(boxes: np.ndarray, output_path: str | Path) -> Path:
    """Write rows shaped as ``x_center, y_center, width, height, class_id``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in np.asarray(boxes).reshape(-1, 5):
        coords = row[:4]
        if not np.isfinite(coords).all():
            raise ValueError(f"Non-finite YOLO coordinates: {coords}")
        if ((coords < 0) | (coords > 1)).any():
            raise ValueError(f"YOLO coordinates must be in [0, 1]: {coords}")
        lines.append(
            f"{int(row[4])} " + " ".join(f"{float(value):.6f}" for value in coords)
        )
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path
