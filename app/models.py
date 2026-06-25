from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from ultralytics import YOLO


BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
YOLO_MODEL_NAME = "yolo11n.pt"


@dataclass(frozen=True)
class ModelBundle:
    device: torch.device
    yolo: YOLO
    blip_processor: BlipProcessor
    blip_model: BlipForConditionalGeneration


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Load heavy ML models once per process and reuse them for every request."""
    device = get_device()

    yolo = YOLO(YOLO_MODEL_NAME)

    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    blip_model.to(device)
    blip_model.eval()

    return ModelBundle(
        device=device,
        yolo=yolo,
        blip_processor=blip_processor,
        blip_model=blip_model,
    )


def models_are_loaded() -> bool:
    return get_model_bundle.cache_info().currsize > 0
