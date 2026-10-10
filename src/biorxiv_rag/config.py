"""Runtime settings read from environment variables.

Configurações de execução lidas das variáveis de ambiente.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAG_MODEL_NAME = "biorxiv-rag"
CLASSIFIER_MODEL_NAME = "biorxiv-area-classifier"
MODEL_ALIAS = "champion"
EXPERIMENT_NAME = "biorxiv-rag"


def _env(name, default=""):
    value = os.environ.get(name, "")
    return value.strip() if value.strip() else default


def _env_int(name, default=None):
    value = _env(name)
    return int(value) if value else default


def _env_bool(name, default):
    value = _env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ArtifactPaths:
    """Files saved by the notebook / arquivos salvos pelo notebook."""

    embedder_dir: Path
    reranker_dir: Path
    embeddings_path: Path
    articles_path: Path
    rag_config_path: Path

    @classmethod
    def from_dir(cls, artifacts_dir):
        root = Path(artifacts_dir)
        return cls(
            embedder_dir=root / "embedder",
            reranker_dir=root / "reranker",
            embeddings_path=root / "bge_large_abstract_embeddings.pt",
            articles_path=root / "articles_full_v2.parquet",
            rag_config_path=root / "rag_config.json",
        )

    def missing(self):
        return [str(path) for path in vars(self).values() if not Path(path).exists()]


def classifier_dir(artifacts_dir):
    return Path(artifacts_dir) / "roberta_classifier_v2" / "best"


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    model_source: str
    mlflow_tracking_uri: str
    mlflow_public_url: str
    rag_model_uri: str
    classifier_model_uri: str
    device: str
    classifier_device: str
    llm_backend: str
    llm_model: str
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    hf_token: str
    n_retrieve: int | None
    n_context: int | None
    max_new_tokens: int | None
    rerank_batch_size: int
    trace_queries: bool

    @classmethod
    def from_env(cls):
        return cls(
            artifacts_dir=Path(_env("ARTIFACTS_DIR", str(REPO_ROOT / "output"))),
            # auto tries the MLflow registry first, then the local folder / auto tenta o registry do MLflow e depois a pasta local
            model_source=_env("MODEL_SOURCE", "auto").lower(),
            mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI"),
            mlflow_public_url=_env("MLFLOW_PUBLIC_URL", "http://localhost:5000"),
            rag_model_uri=_env("RAG_MODEL_URI", f"models:/{RAG_MODEL_NAME}@{MODEL_ALIAS}"),
            classifier_model_uri=_env("CLASSIFIER_MODEL_URI", f"models:/{CLASSIFIER_MODEL_NAME}@{MODEL_ALIAS}"),
            device=_env("DEVICE", "auto").lower(),
            # the classifier stays on the CPU so a small GPU is left for retrieval / o classificador fica na CPU para deixar uma GPU pequena para a recuperação
            classifier_device=_env("CLASSIFIER_DEVICE", "cpu").lower(),
            llm_backend=_env("LLM_BACKEND", "auto").lower(),
            llm_model=_env("LLM_MODEL"),
            llm_provider=_env("LLM_PROVIDER", "auto"),
            llm_base_url=_env("LLM_BASE_URL"),
            llm_api_key=_env("LLM_API_KEY"),
            hf_token=_env("HF_TOKEN"),
            n_retrieve=_env_int("RAG_N_RETRIEVE"),
            n_context=_env_int("RAG_N_CONTEXT"),
            max_new_tokens=_env_int("RAG_MAX_NEW_TOKENS"),
            rerank_batch_size=_env_int("RERANK_BATCH_SIZE", 16),
            trace_queries=_env_bool("MLFLOW_TRACE_QUERIES", True),
        )


def resolve_device(device="auto"):
    """Pick cuda when available / escolher cuda quando disponível."""
    import torch

    if device in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda was requested but no CUDA GPU is visible in this process.")
    return device
