from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.models import get_model_bundle, models_are_loaded
from app.processing import analyze_image
from app.schemas import AnalyzeResponse, HealthResponse


MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_bundle()
    yield


app = FastAPI(
    title="VisionScene AI",
    description="Image scene captioning and object detection API using BLIP and YOLO11n.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    bundle = get_model_bundle()
    return HealthResponse(
        status="ok",
        device=str(bundle.device),
        models_loaded=models_are_loaded(),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum upload size is 5MB.",
        )

    try:
        caption, detections, counts, annotated_image_base64 = analyze_image(
            image_bytes=image_bytes,
            models=get_model_bundle(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return AnalyzeResponse(
        caption=caption,
        detections=detections,
        counts=counts,
        annotated_image_base64=annotated_image_base64,
    )
