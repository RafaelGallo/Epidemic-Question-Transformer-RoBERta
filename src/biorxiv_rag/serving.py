"""Load the pipeline and the classifier for serving, from the MLflow registry or from the local folder.

Carrega o pipeline e o classificador para servir, a partir do registry do MLflow ou da pasta local.
"""

import re
from dataclasses import dataclass

from biorxiv_rag.config import ArtifactPaths, classifier_dir

_ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/@]+)$")


@dataclass
class Loaded:
    """A served model and where it came from / um modelo servido e de onde ele veio."""

    model: object
    source: str
    detail: str
    generator: object = None
    generator_error: str = ""


def _registry_detail(model_uri):
    """Registered version behind a models:/name@alias URI / versão registrada por trás de uma URI models:/nome@alias."""
    from mlflow import MlflowClient

    match = _ALIAS_URI.match(model_uri)
    if not match:
        return model_uri
    version = MlflowClient().get_model_version_by_alias(match["name"], match["alias"])
    return f"{match['name']} v{version.version} @{match['alias']}"


def _use_registry(settings):
    if settings.model_source == "local":
        return False
    if settings.model_source == "mlflow" and not settings.mlflow_tracking_uri:
        raise RuntimeError("MODEL_SOURCE=mlflow needs MLFLOW_TRACKING_URI.")
    return bool(settings.mlflow_tracking_uri)


def _from_registry(settings, model_uri):
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    python_model = mlflow.pyfunc.load_model(model_uri).unwrap_python_model()
    return python_model, _registry_detail(model_uri)


def load_rag(settings):
    """RAG pipeline plus its answer generator / pipeline de RAG e seu gerador de respostas."""
    from biorxiv_rag.llm import try_build_generator
    from biorxiv_rag.pipeline import RagPipeline

    registry_error = ""
    if _use_registry(settings):
        try:
            python_model, detail = _from_registry(settings, settings.rag_model_uri)
            return Loaded(
                python_model.pipeline, "mlflow", detail, python_model.generator, python_model.generator_error
            )
        except Exception as error:
            if settings.model_source == "mlflow":
                raise
            registry_error = f" (MLflow registry unavailable: {error})"

    paths = ArtifactPaths.from_dir(settings.artifacts_dir)
    if paths.missing():
        raise FileNotFoundError(f"Saved RAG files not found: {', '.join(paths.missing())}{registry_error}")
    pipeline = RagPipeline.from_paths(paths, settings.device, settings.rerank_batch_size)
    generator, generator_error = try_build_generator(settings, pipeline)
    return Loaded(pipeline, "local", f"{settings.artifacts_dir}{registry_error}", generator, generator_error)


def load_classifier(settings):
    """Research-area classifier / classificador de área de pesquisa."""
    from biorxiv_rag.classifier import AreaClassifier

    registry_error = ""
    if _use_registry(settings):
        try:
            python_model, detail = _from_registry(settings, settings.classifier_model_uri)
            return Loaded(python_model.classifier, "mlflow", detail)
        except Exception as error:
            if settings.model_source == "mlflow":
                raise
            registry_error = f" (MLflow registry unavailable: {error})"

    model_dir = classifier_dir(settings.artifacts_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Saved classifier not found: {model_dir}{registry_error}")
    return Loaded(AreaClassifier(model_dir, settings.classifier_device), "local", f"{model_dir}{registry_error}")
