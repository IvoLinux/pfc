FROM nvidia/cuda:12.9-base
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# build tools for torch/tokenizers wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
# PROD: DO NOT HOT-RELOAD
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# RELOAD IN DEV
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
