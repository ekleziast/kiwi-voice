# Qwen3-TTS RunPod Serverless Image
# Based on official RunPod PyTorch image with CUDA support

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

LABEL maintainer="ekleziast56"
LABEL version="1.0.0"
LABEL description="Qwen3-TTS voice cloning for RunPod Serverless"

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV MODEL_SIZE=0.6B
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod==1.7.9 \
    transformers==4.57.3 \
    accelerate==1.12.0 \
    librosa \
    soundfile \
    sox \
    einops \
    onnxruntime \
    huggingface_hub

# Install flash-attention (optional, speeds up generation)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Flash attention not available"

# Copy qwen_tts package (as module, not installable package)
COPY qwen_tts /app/qwen_tts

# Copy handler
COPY handler.py /app/handler.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import runpod; print('OK')" || exit 1

# Run handler
CMD ["python", "-u", "/app/handler.py"]
