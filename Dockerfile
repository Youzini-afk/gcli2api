# Multi-stage build for gcli2api
FROM python:3.13-slim as base

ARG TARGETARCH
ARG MIHOMO_VERSION=v1.19.30

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

# Install runtime libraries and the Mihomo core used by the control panel.
# jemalloc 替代 glibc malloc，大幅减少内存碎片化
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl tzdata libjemalloc2 && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    ln -sf "$(ldconfig -p | awk '/libjemalloc.so.2/{print $NF; exit}')" /usr/lib/libjemalloc.so.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mihomo_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$mihomo_arch" in amd64|arm64) ;; *) echo "Unsupported Mihomo architecture: $mihomo_arch"; exit 1 ;; esac; \
    curl -fL "https://github.com/MetaCubeX/mihomo/releases/download/${MIHOMO_VERSION}/mihomo-linux-${mihomo_arch}-${MIHOMO_VERSION}.gz" -o /tmp/mihomo.gz; \
    gzip -dc /tmp/mihomo.gz > /usr/local/bin/mihomo; \
    chmod 0755 /usr/local/bin/mihomo; \
    rm /tmp/mihomo.gz

# 使用 jemalloc 替代 glibc malloc
# jemalloc 会主动将空闲内存归还给操作系统，避免 RSS 持续增长
ENV LD_PRELOAD=/usr/lib/libjemalloc.so.2
# jemalloc 配置：启用后台线程回收、脏页衰减时间 5 秒（默认 10 秒）
ENV MALLOC_CONF="background_thread:true,dirty_decay_ms:5000,muzzy_decay_ms:5000"

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7861

# Default command
CMD ["python", "web.py"]
