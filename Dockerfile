# MLflow Tracking Server
FROM python:3.11-slim AS mlflow

RUN python -m pip install --no-cache-dir \
    mlflow==3.6.0 \
    SQLAlchemy==2.0.57 \
    boto3 \
    psycopg2-binary

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Ray + PyTorch CUDA
FROM rayproject/ray:2.58.0-py311-cpu AS ray

USER root

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 \
    torchvision==0.23.0

RUN python -m pip install --no-cache-dir \
    boto3 \
    mlflow-skinny==3.6.0

COPY app /app

RUN mkdir -p /data/cifar10 /data/results

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

USER ray
