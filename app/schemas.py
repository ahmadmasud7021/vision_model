from typing import Dict, List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left coordinate in pixels")
    y1: int = Field(..., description="Top coordinate in pixels")
    x2: int = Field(..., description="Right coordinate in pixels")
    y2: int = Field(..., description="Bottom coordinate in pixels")


class Detection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class AnalyzeResponse(BaseModel):
    caption: str
    detections: List[Detection]
    counts: Dict[str, int]
    annotated_image_base64: str


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool
