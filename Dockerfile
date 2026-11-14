FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt \
    && groupadd --system --gid 10001 autowrangler \
    && useradd --system --uid 10001 --gid autowrangler --home-dir /home/autowrangler autowrangler \
    && mkdir -p /home/autowrangler \
    && chown -R autowrangler:autowrangler /home/autowrangler

COPY --chown=autowrangler:autowrangler . .

EXPOSE 8501
USER autowrangler

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501"]
