"""MLflow "models from code" entry point for the research-area classifier.

Ponto de entrada "models from code" do MLflow para o classificador de área de pesquisa.
"""

from mlflow.models import set_model

from biorxiv_rag.mlflow_model import ClassifierPyfuncModel

set_model(ClassifierPyfuncModel())
