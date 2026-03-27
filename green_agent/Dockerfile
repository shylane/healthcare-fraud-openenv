# Healthcare Fraud Detection — AgentBeats Green Agent
# Follows canonical AgentBeats pattern (osworld-green, fieldworkarena-green)
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv --no-cache-dir

# Build context must be repo root: docker build -f green_agent/Dockerfile .
COPY green_agent/pyproject.toml pyproject.toml
COPY green_agent/src/ src/

# Environment code is co-located so the green agent can run episodes
COPY environment/ environment/

# Install dependencies (no dev deps in prod image)
RUN uv pip install --system --no-cache .

# Port 9009 is the AgentBeats standard for green agents
EXPOSE 9009

# Entry point matches AgentBeats canonical pattern
ENTRYPOINT ["python", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
