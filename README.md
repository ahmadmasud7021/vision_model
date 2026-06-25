---
title: VisionScene AI
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# VisionScene AI

VisionScene AI is a deployable FastAPI project for image scene understanding. It accepts an uploaded image, preprocesses it with OpenCV, detects objects with YOLO11n, captions the scene with BLIP, and returns a JSON response with an annotated image.

## Features

- `GET /` health check
- `POST /analyze` image analysis endpoint
- Global one-time model loading per API process
- CPU by default, CUDA automatically used when available
- Upload validation for image type and 5MB max size
- OpenCV image decoding, resizing, and annotation
- BLIP scene captioning with `Salesforce/blip-image-captioning-base`
- YOLO object detection with `yolo11n.pt`
- Optional Streamlit frontend in `frontend.py`

## Project Structure

```text
vision_scene_ai/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── processing.py
│   └── schemas.py
├── frontend.py
├── requirements.txt
├── render.yaml
├── README.md
└── .gitignore
```

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first startup can take a while because YOLO and BLIP model weights are downloaded and cached.

## API Usage

Health check:

```bash
curl http://localhost:8000/
```

Analyze an image:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@example.jpg"
```

Example response shape:

```json
{
  "caption": "a person riding a bicycle on a street",
  "detections": [
    {
      "label": "person",
      "confidence": 0.9123,
      "box": {
        "x1": 120,
        "y1": 48,
        "x2": 240,
        "y2": 430
      }
    }
  ],
  "counts": {
    "person": 1
  },
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

## Optional Streamlit Frontend

With the FastAPI server running, start Streamlit:

```bash
streamlit run frontend.py
```

Upload an image, click **Analyze image**, and the frontend will display the generated caption, object counts, and annotated image.

## Render Deployment

This repository includes `render.yaml` for Render Blueprint deployment.

1. Push the project to a GitHub repository.
2. In Render, choose **New +** then **Blueprint**.
3. Select the repository.
4. Render will use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy the service.

For smoother cold starts, use a paid Render plan with enough memory for Torch, Transformers, YOLO, and BLIP. The app uses CPU unless Render provides a CUDA-capable environment.

## Hugging Face Spaces Deployment

For Hugging Face Spaces, create a **Docker Space** and push this repository to the Space git remote.

The included `Dockerfile` runs the Streamlit frontend on port `7860`, which is the default port expected by Spaces. In Spaces, the frontend runs in direct model mode and calls the local YOLO/BLIP pipeline without needing a separate FastAPI URL:

```bash
streamlit run frontend.py --server.address 0.0.0.0 --server.port 7860
```

After creating a Docker Space on Hugging Face, add it as a second git remote:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/vision-scene-ai
git push space main
```

Your interactive demo will be available at:

```text
https://YOUR_USERNAME-vision-scene-ai.hf.space/
```

If you want to deploy the FastAPI API instead of the visual Streamlit demo, change the Docker `CMD` back to:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Notes

- `opencv-python-headless` is used for server deployments.
- Uploaded images are resized to a maximum dimension of 640 pixels while preserving aspect ratio.
- YOLO detection runs with `imgsz=640` and confidence threshold `0.35`.
- The annotated image is returned as a base64-encoded JPEG string.
