FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /srv

# System deps: libgomp for paddle, dejavu fonts for the synthetic generator
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY validators ./validators
COPY demo ./demo
COPY scripts ./scripts

# Golden PNGs are generated, not tracked: produce them inside the image
RUN python scripts/generate_golden.py
COPY data/golden/ground_truth.json ./data/golden/ground_truth.json

# Bake the weights so the container runs fully offline
ENV CEKDULU_MODEL_DIR=/srv/models
RUN python -c "import app.detect as d; d.load_model()" \
    && python -c "import app.detect as d; d.ocr_lines('data/golden/ktp_000.png')" \
    && python -c "import app.face as f; f.load_model()"

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
