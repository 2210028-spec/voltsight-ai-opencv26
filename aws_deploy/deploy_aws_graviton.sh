#!/usr/bin/env bash
# One-Command AWS Free Tier / Graviton (Arm64) Deployment Script for VoltSight AI
set -euo pipefail

export AWS_REGION="${AWS_REGION:-us-east-1}"
export VOLTSIGHT_S3_BUCKET="voltsight-opencv26-$(date +%s)"

echo "[1/3] Creating Amazon S3 evidence bucket: s3://${VOLTSIGHT_S3_BUCKET}"
aws s3 mb "s3://${VOLTSIGHT_S3_BUCKET}" --region "${AWS_REGION}" || true

echo "[2/3] Running reproducible 60-frame OpenCV + Agentic Vision benchmark suite..."
python3 evaluate_benchmarks.py

echo "[3/3] Starting VoltSight AI FastAPI server on port 8080..."
uvicorn app:app --host 0.0.0.0 --port 8080
