# ==========================================
# STAGE 1: Dependency Builder
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Prevent Python from writing pyc files and buffer streams
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install compiler utilities required for certain packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and copy requirements
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .

# Install dependencies into local folder for copying
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# STAGE 2: Lightweight Final Runtime
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Install curl for container orchestration healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user and group for container security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy installed site-packages and binaries from builder
COPY --from=builder /root/.local /home/appuser/.local
RUN chown -R appuser:appgroup /home/appuser

# Copy application codes and model assets (preserving project structure)
COPY --chown=appuser:appgroup src /app/src
COPY --chown=appuser:appgroup tests /app/tests
COPY --chown=appuser:appgroup .streamlit /app/.streamlit

# Switch environment to non-root security context
USER appuser

# Expose Streamlit default hosting port
EXPOSE 8501

# Add standard Container health check checking port status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch application
ENTRYPOINT ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
