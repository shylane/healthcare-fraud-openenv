# Healthcare Fraud Detection — AgentBeats Purple Agent
# Serves the GSPO-trained Qwen3-1.7B + LoRA adapter
#
# Build: docker build -t healthcare-fraud-purple:latest -f purple_agent/Dockerfile .
# Run:   docker run --gpus all -p 9009:9009 -v ./checkpoints/best_adapter:/app/model healthcare-fraud-purple:latest
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.12 python3-pip python3.12-venv git && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install uv --no-cache-dir

# Build context must be repo root: docker build -f purple_agent/Dockerfile .
COPY purple_agent/pyproject.toml pyproject.toml
COPY purple_agent/src/ src/

# Model weights are mounted at runtime (not baked into image — too large)
# Mount: -v ./checkpoints/best_adapter:/app/model
# Or set MODEL_PATH env var to a HuggingFace model ID for on-demand download
ENV MODEL_PATH=/app/model

RUN uv pip install --system --no-cache \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    a2a-sdk>=0.2.0 \
    uvicorn>=0.30.0 \
    httpx>=0.27.0 \
    pydantic>=2.0.0

EXPOSE 9009

ENTRYPOINT ["python3", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009", "--model-path", "/app/model"]
