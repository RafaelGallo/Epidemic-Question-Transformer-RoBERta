# syntax=docker/dockerfile:1
# One image for the three services: MLflow server, model registration and the Streamlit app.
# Uma imagem para os três serviços: servidor MLflow, registro dos modelos e app Streamlit.
FROM python:3.13-slim

# CPU wheels by default; the GPU compose file switches the index to a CUDA build
# wheels de CPU por padrão; o compose de GPU troca o índice para um build CUDA
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# the volumes mounted on these folders inherit this ownership / os volumes montados nestas pastas herdam este dono
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app /mlflow/db /mlflow/artifacts /home/app/.cache/huggingface \
    && chown -R app:app /app /mlflow /home/app

WORKDIR /app
COPY --chown=app:app .streamlit/ .streamlit/
COPY --chown=app:app src/ src/
COPY --chown=app:app mlflow_models/ mlflow_models/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app app/ app/
COPY --chown=app:app main.py ./

ENV PYTHONPATH=/app/src \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

USER app
EXPOSE 8501 5000

CMD ["streamlit", "run", "app/streamlit_app.py"]
