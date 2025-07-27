# ── 1. Use a ready-made RAPIDS image ──────────────────────────────────────
FROM rapidsai/notebooks:25.04a-cuda12.8-py3.12

USER root

# ── 2. Install anything RAPIDS doesn’t ship with (small) ──────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git  curl && \
    rm -rf /var/lib/apt/lists/*

# Python bits: Kaggle CLI + light viz libs that are NOT in the base image
RUN pip install --no-cache-dir \
        kaggle==1.6.12 \
        imbalanced-learn==0.12.* \
        seaborn==0.13.* \
        matplotlib==3.9.* \
        joblib==1.4.*

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

USER rapids
WORKDIR /home/rapids

EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/start.sh"]