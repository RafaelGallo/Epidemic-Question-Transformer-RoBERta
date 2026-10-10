"""MLflow pyfunc wrappers for the RAG pipeline and the area classifier.

Wrappers pyfunc do MLflow para o pipeline de RAG e o classificador de área.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from mlflow.pyfunc import PythonModel

from biorxiv_rag.config import ArtifactPaths, Settings


def _column(model_input, name):
    """Values of one input column, whatever container MLflow hands over / valores de uma coluna de entrada, em qualquer formato que o MLflow entregar."""
    if isinstance(model_input, pd.DataFrame):
        return model_input[name].astype(str).tolist()
    if isinstance(model_input, dict):
        values = model_input[name]
        return [str(values)] if isinstance(values, str) else [str(value) for value in values]
    return [str(row[name]) if isinstance(row, dict) else str(row) for row in model_input]


class RagPyfuncModel(PythonModel):
    """Question in, grounded answer and cited sources out / entra a pergunta, saem a resposta ancorada e as fontes citadas."""

    def load_context(self, context):
        from biorxiv_rag.llm import try_build_generator
        from biorxiv_rag.pipeline import RagPipeline

        settings = Settings.from_env()
        artifacts = context.artifacts
        paths = ArtifactPaths(
            embedder_dir=Path(artifacts["embedder"]),
            reranker_dir=Path(artifacts["reranker"]),
            embeddings_path=Path(artifacts["embeddings"]),
            articles_path=Path(artifacts["articles"]),
            rag_config_path=Path(artifacts["rag_config"]),
        )
        self.pipeline = RagPipeline.from_paths(paths, settings.device, settings.rerank_batch_size)
        self.generator, self.generator_error = try_build_generator(settings, self.pipeline)

    def predict(self, context, model_input, params=None):
        params = params or {}
        rows = []
        for question in _column(model_input, "question"):
            result = self.pipeline.search(
                question,
                n_retrieve=int(params.get("n_retrieve") or 0) or None,
                n_context=int(params.get("n_context") or 0) or None,
                area=params.get("area") or None,
                dedupe=bool(params.get("dedupe", True)),
            )
            answer = ""
            if self.generator is not None and params.get("generate", True):
                generation = self.generator.start(
                    self.pipeline.build_messages(question, result.sources),
                    int(params.get("max_new_tokens") or self.pipeline.rag_config["max_new_tokens"]),
                )
                answer = "".join(generation).strip()
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": json.dumps([asdict(source) for source in result.sources], ensure_ascii=False),
                }
            )
        return pd.DataFrame(rows)


class ClassifierPyfuncModel(PythonModel):
    """Title and abstract in, most likely research areas out / entram título e resumo, saem as áreas mais prováveis."""

    def load_context(self, context):
        from biorxiv_rag.classifier import AreaClassifier

        self.classifier = AreaClassifier(context.artifacts["model"], Settings.from_env().classifier_device)

    def predict(self, context, model_input, params=None):
        top_k = int((params or {}).get("top_k") or 3)
        titles = _column(model_input, "title")
        abstracts = _column(model_input, "abstract")
        probabilities = self.classifier.predict_proba(titles, abstracts)
        rows = []
        for probs in probabilities:
            best = probs.argsort()[::-1][:top_k]
            rows.append(
                {
                    "area": self.classifier.area_names[best[0]],
                    "probability": float(probs[best[0]]),
                    "top_areas": json.dumps(
                        [{"area": self.classifier.area_names[i], "probability": float(probs[i])} for i in best]
                    ),
                }
            )
        return pd.DataFrame(rows)
