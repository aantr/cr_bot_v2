import cv2
import time
from pathlib import Path

from ultralytics import YOLO

from model_paths import BATTLEFIELDS, DETECTION_ENGINE_PATH

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = DETECTION_ENGINE_PATH

INPUT_VIDEO = "screenshots\\last_20_percent.mp4"
OUTPUT_VIDEO = "screenshots\\output_detected.mp4"

IMGSZ = 1280

CONF = 0.50
IOU = 0.50

MAX_DET = 500

DEVICE = 0

# Для .pt можно оставить True.
# Для TensorRT .engine можно поставить False,
# потому что FP16 уже задаётся при экспорте engine.
QUANTIZE = 16  # FP16; use None for FP32


# =========================================================
# BATTLEFIELD CROPS
# =========================================================
#
# Ключ:
#
#     (width, height)
#
# Значение:
#
#     (x1, y1, x2, y2)
#
# Координаты относятся к исходному изображению видео.
#
# ВАЖНО:
# OpenCV возвращает размер как:
#
#     width x height
#
# Поэтому ключ именно:
#
#     (width, height)
#
# =========================================================

# =========================================================
# LOAD MODEL
# =========================================================

print()
print("=" * 60)
print("Loading YOLO model")
print("=" * 60)

print(f"Model: {MODEL_PATH}")

model = YOLO(
    MODEL_PATH
)


# =========================================================
# OPEN VIDEO
# =========================================================

print()
print("=" * 60)
print("Opening video")
print("=" * 60)

print(f"Input: {INPUT_VIDEO}")


cap = cv2.VideoCapture(
    INPUT_VIDEO
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {INPUT_VIDEO}"
    )


# =========================================================
# VIDEO INFO
# =========================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print()
print("Video info:")
print(f"Resolution: {width}x{height}")
print(f"FPS:        {fps:.2f}")
print(f"Frames:     {total_frames}")
print(f"Duration:   {duration:.2f} sec")


# =========================================================
# SELECT BATTLEFIELD CROP
# =========================================================

video_size = (
    width,
    height,
)


if video_size not in BATTLEFIELDS:

    raise ValueError(
        "\n"
        f"Unsupported video resolution: "
        f"{width}x{height}\n"
        "\n"
        f"Add crop coordinates to BATTLEFIELDS in model_paths.py:\n"
        "\n"
        f"BATTLEFIELDS = {{\n"
        f"    ({width}, {height}): "
        f"(x1, y1, x2, y2),\n"
        f"}}\n"
    )


BATTLEFIELD = BATTLEFIELDS[
    video_size
]


(
    crop_x1,
    crop_y1,
    crop_x2,
    crop_y2,
) = BATTLEFIELD


# =========================================================
# VALIDATE CROP
# =========================================================

if crop_x1 < 0:

    raise ValueError(
        "crop_x1 cannot be < 0"
    )


if crop_y1 < 0:

    raise ValueError(
        "crop_y1 cannot be < 0"
    )


if crop_x2 > width:

    raise ValueError(
        f"crop_x2 ({crop_x2}) "
        f"is larger than video width ({width})"
    )


if crop_y2 > height:

    raise ValueError(
        f"crop_y2 ({crop_y2}) "
        f"is larger than video height ({height})"
    )


if crop_x2 <= crop_x1:

    raise ValueError(
        "crop_x2 must be larger than crop_x1"
    )


if crop_y2 <= crop_y1:

    raise ValueError(
        "crop_y2 must be larger than crop_y1"
    )


battlefield_width = (
    crop_x2 - crop_x1
)

battlefield_height = (
    crop_y2 - crop_y1
)


print()
print("Battlefield crop:")

print(
    f"x1={crop_x1}, "
    f"y1={crop_y1}, "
    f"x2={crop_x2}, "
    f"y2={crop_y2}"
)

print(
    f"Battlefield size: "
    f"{battlefield_width}x{battlefield_height}"
)


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

output_path = Path(
    OUTPUT_VIDEO
)


output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# OUTPUT VIDEO WRITER
# =========================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (
        width,
        height,
    ),
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create output video: "
        f"{OUTPUT_VIDEO}"
    )


# =========================================================
# PROCESS VARIABLES
# =========================================================

frame_index = 0

total_inference_time = 0.0

processing_start = (
    time.perf_counter()
)


# =========================================================
# PROCESS VIDEO
# =========================================================

print()
print("=" * 60)
print("Processing")
print("=" * 60)
print()


while True:

    # -----------------------------------------------------
    # READ FRAME
    # -----------------------------------------------------

    ok, frame = cap.read()


    if not ok:
        break


    frame_index += 1


    # =====================================================
    # CROP BATTLEFIELD
    # =====================================================

    battlefield = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    # =====================================================
    # YOLO INFERENCE
    # =====================================================

    inference_start = (
        time.perf_counter()
    )


    results = model.predict(

        source=battlefield,

        imgsz=IMGSZ,

        conf=CONF,

        iou=IOU,

        max_det=MAX_DET,

        device=DEVICE,

        quantize=QUANTIZE,

        verbose=False,
    )


    inference_end = (
        time.perf_counter()
    )


    inference_time = (
        inference_end
        -
        inference_start
    )


    total_inference_time += (
        inference_time
    )


    result = results[0]


    # =====================================================
    # DETECTIONS
    # =====================================================

    detection_count = 0


    if result.boxes is not None:

        detection_count = len(
            result.boxes
        )


        for box in result.boxes:

            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            class_name = (
                model.names[
                    class_id
                ]
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # BBOX IN CROPPED IMAGE
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            # ------------------------------------------------
            # CONVERT TO FULL FRAME COORDINATES
            # ------------------------------------------------

            full_x1 = (
                x1
                +
                crop_x1
            )


            full_y1 = (
                y1
                +
                crop_y1
            )


            full_x2 = (
                x2
                +
                crop_x1
            )


            full_y2 = (
                y2
                +
                crop_y1
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = (
                full_x1
                +
                full_x2
            ) // 2


            center_y = (
                full_y1
                +
                full_y2
            ) // 2


            # =================================================
            # HERE YOU CAN USE DETECTIONS
            # =================================================
            #
            # Например:
            #
            # if class_name == "skeleton":
            #
            #     print(
            #         f"Frame={frame_index} "
            #         f"class={class_name} "
            #         f"conf={confidence:.2f} "
            #         f"center=({center_x}, {center_y})"
            #     )
            #
            # =================================================


    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    detected_battlefield = (
        result.plot()
    )


    # =====================================================
    # PUT CROPPED IMAGE BACK INTO FULL FRAME
    # =====================================================

    output_frame = (
        frame.copy()
    )


    output_frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ] = detected_battlefield


    # =====================================================
    # INFERENCE FPS
    # =====================================================

    if inference_time > 0:

        inference_fps = (
            1.0
            /
            inference_time
        )

    else:

        inference_fps = 0


    # =====================================================
    # PROGRESS
    # =====================================================

    if total_frames > 0:

        progress = (
            frame_index
            /
            total_frames
            *
            100
        )

    else:

        progress = 0


    # =====================================================
    # DRAW DEBUG INFO
    # =====================================================

    cv2.putText(
        output_frame,

        (
            f"Inference: "
            f"{inference_fps:.1f} FPS"
        ),

        (
            20,
            40,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (
            255,
            255,
            255,
        ),

        2,

        cv2.LINE_AA,
    )


    cv2.putText(
        output_frame,

        (
            f"Frame: "
            f"{frame_index}/{total_frames}"
        ),

        (
            20,
            80,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            255,
            255,
            255,
        ),

        2,

        cv2.LINE_AA,
    )


    cv2.putText(
        output_frame,

        (
            f"Objects: "
            f"{detection_count}"
        ),

        (
            20,
            115,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            255,
            255,
            255,
        ),

        2,

        cv2.LINE_AA,
    )


    # =====================================================
    # WRITE FRAME
    # =====================================================

    writer.write(
        output_frame
    )


    # =====================================================
    # CONSOLE PROGRESS
    # =====================================================

    if (
        frame_index % 30 == 0
        or
        frame_index == total_frames
    ):

        if frame_index > 0:

            avg_inference = (
                total_inference_time
                /
                frame_index
            )

        else:

            avg_inference = 0


        if avg_inference > 0:

            avg_fps = (
                1.0
                /
                avg_inference
            )

        else:

            avg_fps = 0


        elapsed = (
            time.perf_counter()
            -
            processing_start
        )


        if frame_index > 0:

            processing_fps = (
                frame_index
                /
                elapsed
            )

        else:

            processing_fps = 0


        if processing_fps > 0:

            frames_left = (
                total_frames
                -
                frame_index
            )

            eta = (
                frames_left
                /
                processing_fps
            )

        else:

            eta = 0


        print(
            "\r"
            f"{progress:6.2f}% | "
            f"{frame_index}/{total_frames} | "
            f"detections={detection_count:3d} | "
            f"inference={inference_fps:6.1f} FPS | "
            f"avg={avg_fps:6.1f} FPS | "
            f"ETA={eta:6.1f}s",
            end="",
            flush=True,
        )


# =========================================================
# CLEANUP
# =========================================================

cap.release()

writer.release()


# =========================================================
# FINAL STATISTICS
# =========================================================

total_processing_time = (
    time.perf_counter()
    -
    processing_start
)


print()
print()
print("=" * 60)
print("Done")
print("=" * 60)

print(
    f"Output: "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Processed frames: "
    f"{frame_index}"
)

print(
    f"Processing time: "
    f"{total_processing_time:.2f} sec"
)


if frame_index > 0:

    overall_processing_fps = (
        frame_index
        /
        total_processing_time
    )

    print(
        f"Overall processing FPS: "
        f"{overall_processing_fps:.2f}"
    )


if (
    total_inference_time > 0
    and
    frame_index > 0
):

    avg_inference_ms = (
        total_inference_time
        /
        frame_index
        *
        1000
    )


    avg_inference_fps = (
        frame_index
        /
        total_inference_time
    )


    print(
        f"Average inference: "
        f"{avg_inference_ms:.2f} ms/frame"
    )


    print(
        f"Average inference FPS: "
        f"{avg_inference_fps:.2f}"
    )
