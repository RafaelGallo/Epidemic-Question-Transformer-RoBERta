"""MLflow "models from code" entry point for the RAG pipeline.

Ponto de entrada "models from code" do MLflow para o pipeline de RAG.
"""

from mlflow.models import set_model

from biorxiv_rag.mlflow_model import RagPyfuncModel

set_model(RagPyfuncModel())
