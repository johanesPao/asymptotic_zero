# ─── Stage 1: Build TA-Lib C library ───────────────────────────────────────
FROM python:3.13-slim AS talib-builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make wget ca-certificates autoconf libtool pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Download and build TA-Lib C library
RUN wget -O ta-lib.tar.gz \
        https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make  \
    && make install \
    && cd / \
    && rm -rf ta-lib ta-lib.tar.gz

# ─── Stage 2: Python dependencies ──────────────────────────────────────────
FROM python:3.13-slim AS deps-builder

# Copy compiled TA-Lib C library from stage 1
COPY --from=talib-builder /usr/lib/libta_lib* /usr/lib/
COPY --from=talib-builder /usr/include/ta-lib /usr/include/ta-lib

# Install build tools for Python packages that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-prod.txt /tmp/requirements-prod.txt

# Install Python dependencies to a custom prefix (to copy into final image)
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements-prod.txt

# ─── Stage 3: Runtime image ─────────────────────────────────────────────────
FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/johanesPao/asymptotic_zero"
LABEL org.opencontainers.image.description="Asymptotic Zero — DQN crypto futures trading bot"

# Copy TA-Lib C library from stage 1
COPY --from=talib-builder /usr/lib/libta_lib* /usr/lib/
COPY --from=talib-builder /usr/include/ta-lib /usr/include/ta-lib
RUN ldconfig

# Copy installed Python packages
COPY --from=deps-builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy project source — .dockerignore handles exclusions
COPY . .

# Ensure checkpoints/best exists
RUN test -d checkpoints/best || (echo "ERROR: checkpoints/best not found. Commit the trained model first." && exit 1)

# Non-root user for security
RUN useradd -m -u 1000 az && chown -R az:az /app
USER az

# Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Entry point
CMD ["python", "scripts/trading_bot.py"]
