FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PULMOAI_USER_DATA_DIR=/var/lib/pulmoai MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-runtime.txt
COPY backend/ backend/
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY reports/ reports/
COPY samples/ samples/
COPY --from=frontend /build/frontend/dist/ frontend/dist/
RUN useradd --uid 10001 --create-home pulmoai && mkdir -p /var/lib/pulmoai && chown -R pulmoai:pulmoai /var/lib/pulmoai
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=4)"
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
