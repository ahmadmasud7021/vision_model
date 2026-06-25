import base64
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from app.models import ModelBundle
from app.schemas import Detection


MAX_IMAGE_DIMENSION = 640
YOLO_CONFIDENCE_THRESHOLD = 0.35


def decode_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file could not be decoded as an image.")
    return image


def resize_keep_aspect_ratio(image: np.ndarray, max_dimension: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    height, width = image.shape[:2]
    largest_side = max(height, width)

    if largest_side <= max_dimension:
        return image.copy()

    scale = max_dimension / largest_side
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def generate_caption(image_bgr: np.ndarray, models: ModelBundle) -> str:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = models.blip_processor(images=pil_image, return_tensors="pt")
    inputs = {key: value.to(models.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = models.blip_model.generate(**inputs, max_new_tokens=40)

    caption = models.blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def detect_objects(image_bgr: np.ndarray, models: ModelBundle) -> List[Detection]:
    results = models.yolo.predict(
        source=image_bgr,
        imgsz=MAX_IMAGE_DIMENSION,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
        device=str(models.device),
    )

    detections: List[Detection] = []
    if not results:
        return detections

    result = results[0]
    names = result.names

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy()
        confidence = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())
        label = str(names.get(class_id, class_id))

        x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
        detections.append(
            Detection(
                label=label,
                confidence=round(confidence, 4),
                box={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            )
        )

    return detections


def count_objects(detections: List[Detection]) -> Dict[str, int]:
    return dict(Counter(detection.label for detection in detections))


def draw_annotations(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()

    for detection in detections:
        box = detection.box
        label = f"{detection.label} {detection.confidence:.2f}"

        color = _color_for_label(detection.label)
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)

        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        text_x = max(box.x1, 0)
        text_y = max(box.y1 - 8, text_height + baseline + 4)

        cv2.rectangle(
            annotated,
            (text_x, text_y - text_height - baseline - 4),
            (text_x + text_width + 6, text_y + baseline - 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (text_x + 3, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def encode_image_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise ValueError("Annotated image could not be encoded.")
    return base64.b64encode(buffer).decode("utf-8")


def analyze_image(image_bytes: bytes, models: ModelBundle) -> Tuple[str, List[Detection], Dict[str, int], str]:
    image = decode_image(image_bytes)
    resized = resize_keep_aspect_ratio(image)

    caption = generate_caption(resized, models)
    detections = detect_objects(resized, models)
    counts = count_objects(detections)
    annotated = draw_annotations(resized, detections)
    annotated_base64 = encode_image_base64(annotated)

    return caption, detections, counts, annotated_base64


def _color_for_label(label: str) -> Tuple[int, int, int]:
    seed = sum(ord(char) for char in label)
    return (
        60 + (seed * 37) % 180,
        60 + (seed * 17) % 180,
        60 + (seed * 29) % 180,
    )
