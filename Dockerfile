# Multi-stage Dockerfile for UCMEC-mmWave-Fronthaul Multi-Objective Platform

# Stage 1: Build Frontend Static Bundle
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Simulator Runtime
FROM python:3.10-slim-bookworm

# System dependencies for scientific computing and conic solvers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source code
COPY . .

# Copy built frontend assets
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose API and frontend dashboard port
EXPOSE 8000

# Persistent storage directories
VOLUME ["/app/experiments_storage", "/app/results"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
