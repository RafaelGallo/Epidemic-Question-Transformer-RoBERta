"""Register the saved RAG pipeline and the classifier in the MLflow Model Registry.

Registra o pipeline de RAG e o classificador salvos no Model Registry do MLflow.
Copia os arquivos uma única vez: se nada mudou, o registro existente é mantido.

    python scripts/register_models.py [--artifacts-dir output] [--force]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

# make src/ importable without installing the package / tornar src/ importável sem instalar o pacote
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

from biorxiv_rag import __version__, notebook_results
from biorxiv_rag.config import (
    CLASSIFIER_MODEL_NAME,
    EXPERIMENT_NAME,
    MODEL_ALIAS,
    RAG_MODEL_NAME,
    REPO_ROOT,
    ArtifactPaths,
    Settings,
    classifier_dir,
)
from biorxiv_rag.corpus import load_rag_config

EDGE_BYTES = 1024 * 1024
SERVING_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy", "pandas", "pyarrow", "mlflow")
IMPORT_TAGS = {
    "origin": "notebook/Transformer_RoBERta.ipynb (Google Colab)",
    "metrics_origin": "copied from the notebook outputs, not re-executed",
    "package_version": __version__,
}


def fingerprint(paths):
    """Cheap content hash: file names, sizes and the first and last megabyte of each file.

    Hash barato do conteúdo: nomes, tamanhos e o primeiro e o último megabyte de cada arquivo.
    """
    digest = hashlib.sha256(__version__.encode())
    for path in paths:
        path = Path(path)
        files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        for file in files:
            size = file.stat().st_size
            digest.update(f"{file.name}:{size}".encode())
            with open(file, "rb") as f:
                digest.update(f.read(EDGE_BYTES))
                if size > 2 * EDGE_BYTES:
                    f.seek(-EDGE_BYTES, os.SEEK_END)
                    digest.update(f.read(EDGE_BYTES))
    return digest.hexdigest()[:16]


def champion_fingerprint(client, name):
    try:
        return client.get_model_version_by_alias(name, MODEL_ALIAS).tags.get("fingerprint", "")
    except MlflowException:
        return ""


def pip_requirements():
    # the local version tag (+cpu, +cu126) is not installable from PyPI / a tag local (+cpu, +cu126) não é instalável pelo PyPI
    return [f"{name}=={version(name).split('+')[0]}" for name in SERVING_PACKAGES]


def promote(client, name, model_version, mark, description):
    client.set_model_version_tag(name, str(model_version), "fingerprint", mark)
    client.set_registered_model_alias(name, MODEL_ALIAS, str(model_version))
    client.update_registered_model(name, description=description)
    print(f"{name}: version {model_version} is now @{MODEL_ALIAS}")


def log_model(entry_point, artifacts, signature, registered_name):
    started = time.perf_counter()
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=str(REPO_ROOT / "mlflow_models" / entry_point),
        artifacts={key: str(path) for key, path in artifacts.items()},
        code_paths=[str(REPO_ROOT / "src" / "biorxiv_rag")],
        pip_requirements=pip_requirements(),
        signature=signature,
        registered_model_name=registered_name,
    )
    print(f"{registered_name}: files copied to the artifact store in {time.perf_counter() - started:.0f}s")
    return info


def register_rag(client, artifacts_dir, force):
    paths = ArtifactPaths.from_dir(artifacts_dir)
    if paths.missing():
        raise FileNotFoundError(f"Saved RAG files not found: {', '.join(paths.missing())}")
    artifacts = {
        "embedder": paths.embedder_dir,
        "reranker": paths.reranker_dir,
        "embeddings": paths.embeddings_path,
        "articles": paths.articles_path,
        "rag_config": paths.rag_config_path,
    }
    mark = fingerprint(artifacts.values())
    if not force and champion_fingerprint(client, RAG_MODEL_NAME) == mark:
        print(f"{RAG_MODEL_NAME}: already registered with the same files, nothing to do")
        return

    rag_config = load_rag_config(paths.rag_config_path)
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "question")]),
        outputs=Schema([ColSpec("string", "question"), ColSpec("string", "answer"), ColSpec("string", "sources")]),
        params=ParamSchema(
            [
                ParamSpec("n_retrieve", "long", int(rag_config["n_retrieve"])),
                ParamSpec("n_context", "long", int(rag_config["n_context"])),
                ParamSpec("max_new_tokens", "long", int(rag_config["max_new_tokens"])),
                ParamSpec("area", "string", ""),
                ParamSpec("dedupe", "boolean", True),
                ParamSpec("generate", "boolean", True),
            ]
        ),
    )
    with mlflow.start_run(run_name="rag-pipeline"):
        mlflow.set_tags(IMPORT_TAGS)
        mlflow.log_params(notebook_results.rag_params(rag_config))
        mlflow.log_metrics(notebook_results.rag_metrics())
        mlflow.log_artifact(str(paths.rag_config_path), "notebook_outputs")
        answers_path = Path(artifacts_dir) / "rag_answers.csv"
        if answers_path.exists():
            mlflow.log_artifact(str(answers_path), "notebook_outputs")
        info = log_model("rag_model.py", artifacts, signature, RAG_MODEL_NAME)
    promote(
        client,
        RAG_MODEL_NAME,
        info.registered_model_version,
        mark,
        "bioRxiv RAG: bge-large-en-v1.5 dense retrieval over 99,540 abstracts, bge-reranker-large reranking "
        "and a grounded answer with cited sources.",
    )


def training_history(artifacts_dir):
    """Loss and validation curves saved by the Trainer / curvas de loss e validação salvas pelo Trainer."""
    states = sorted((Path(artifacts_dir) / "roberta_classifier_v2" / "checkpoints").glob("*/trainer_state.json"))
    if not states:
        return []
    with open(states[-1], "r", encoding="utf-8") as f:
        log_history = json.load(f)["log_history"]
    now = int(time.time() * 1000)
    names = {
        "loss": "train_loss",
        "learning_rate": "learning_rate",
        "eval_loss": "val_loss",
        "eval_accuracy": "val_accuracy",
        "eval_f1_macro": "val_f1_macro",
    }
    return [
        Metric(names[key], float(value), now, int(entry["step"]))
        for entry in log_history
        for key, value in entry.items()
        if key in names
    ]


def register_classifier(client, artifacts_dir, figures_dir, force):
    model_dir = classifier_dir(artifacts_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Saved classifier not found: {model_dir}")
    mark = fingerprint([model_dir])
    if not force and champion_fingerprint(client, CLASSIFIER_MODEL_NAME) == mark:
        print(f"{CLASSIFIER_MODEL_NAME}: already registered with the same files, nothing to do")
        return

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "title"), ColSpec("string", "abstract")]),
        outputs=Schema([ColSpec("string", "area"), ColSpec("double", "probability"), ColSpec("string", "top_areas")]),
        params=ParamSchema([ParamSpec("top_k", "long", 3)]),
    )
    with mlflow.start_run(run_name="roberta-area-classifier") as run:
        mlflow.set_tags(IMPORT_TAGS)
        mlflow.log_params(notebook_results.CLASSIFIER_PARAMS)
        mlflow.log_metrics(notebook_results.CLASSIFIER_TEST_METRICS)
        history = training_history(artifacts_dir)
        if history:
            client.log_batch(run.info.run_id, metrics=history)
        # figures and test predictions produced by the notebook / figuras e predições de teste geradas pelo notebook
        for figure in sorted(Path(figures_dir).glob("*.png")):
            mlflow.log_artifact(str(figure), "figures")
        predictions_path = Path(artifacts_dir) / "predictions_e2e.csv"
        if predictions_path.exists():
            mlflow.log_artifact(str(predictions_path), "notebook_outputs")
        info = log_model("classifier_model.py", {"model": model_dir}, signature, CLASSIFIER_MODEL_NAME)
    promote(
        client,
        CLASSIFIER_MODEL_NAME,
        info.registered_model_version,
        mark,
        "RoBERTa-large fine-tuned on bioRxiv preprints: research area (25 classes) from title and abstract.",
    )


def main():
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts-dir", default=str(settings.artifacts_dir))
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "img"))
    parser.add_argument("--force", action="store_true", help="register a new version even if the files did not change")
    args = parser.parse_args()

    if not settings.mlflow_tracking_uri:
        sys.exit("MLFLOW_TRACKING_URI is not set. Example: MLFLOW_TRACKING_URI=http://localhost:5000")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    # a real API call: /health also answers when the Host header is rejected / uma chamada real de API: o /health responde mesmo quando o header Host é rejeitado
    client.search_experiments(max_results=1)
    mlflow.set_experiment(EXPERIMENT_NAME)

    register_rag(client, args.artifacts_dir, args.force)
    register_classifier(client, args.artifacts_dir, args.figures_dir, args.force)


if __name__ == "__main__":
    main()
