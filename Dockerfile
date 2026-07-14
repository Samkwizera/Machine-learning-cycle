# Streamlit UI image (the user-facing deployment package, e.g. on Render).
FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch first (keeps the image small vs the default CUDA build).
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code, shared modules, trained model, viz cache, theme.
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/viz_cache.json data/viz_cache.json
COPY .streamlit/ .streamlit/

EXPOSE 8501
ENV PORT=8501

# Render/most PaaS inject $PORT; default to 8501 locally.
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
