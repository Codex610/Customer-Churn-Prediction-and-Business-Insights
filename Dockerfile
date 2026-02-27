# ─── Base image ───────────────────────────────────────────
FROM python:3.10-slim

# ─── Set working directory ────────────────────────────────
WORKDIR /app

# ─── Install system dependencies ─────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ─── Copy and install Python dependencies ────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy source code and models ─────────────────────────
COPY src/         ./src/
COPY models/      ./models/
COPY src/config.py ./config.py

# ─── Expose FastAPI port ──────────────────────────────────
EXPOSE 8000

# ─── Run the API ─────────────────────────────────────────
CMD ["uvicorn", "src.deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]