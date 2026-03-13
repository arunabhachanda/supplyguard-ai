# ── SupplyGuard AI — Multi-service Docker Image ───────────────────
# Runs Streamlit (port 8501) + FastAPI (port 8001) with supervisord

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create non-root user (security hardening)
RUN useradd -m -u 1001 supplyguard \
    && chown -R supplyguard:supplyguard /app
USER supplyguard

# ── Supervisord config ────────────────────────────────────────────
USER root
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor

COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ── Expose ports ──────────────────────────────────────────────────
# 8501: Streamlit UI
# 8001: FastAPI REST
EXPOSE 8501 8001

# ── Health check ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
