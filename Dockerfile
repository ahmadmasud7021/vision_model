FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV VISIONSCENE_DIRECT_MODE=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend.py .

EXPOSE 7860

CMD ["streamlit", "run", "frontend.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
