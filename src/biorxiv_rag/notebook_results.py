"""Numbers reported by notebook/Transformer_RoBERta.ipynb, copied from its cell outputs.

Números reportados pelo notebook, copiados das saídas das células. Nada aqui é recalculado no deploy.
"""

DATASET = "abuzarmahmood/bioarxiv-xml-100k"
DATASET_URL = "https://www.kaggle.com/datasets/abuzarmahmood/bioarxiv-xml-100k"
TRAINING_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

# Part 7 and Part 9 of the notebook / Partes 7 e 9 do notebook
CLASSIFIER_PARAMS = {
    "base_model": "deepset/roberta-large-squad2",
    "num_labels": 25,
    "max_length": 384,
    "truncation": "only_second",
    "input": "title + abstract (text pair)",
    "num_train_epochs": 3,
    "train_batch_size": 32,
    "eval_batch_size": 128,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 510,
    "precision": "bf16",
    "seed": 42,
    "metric_for_best_model": "f1_macro",
    "total_steps": 5106,
    "max_train_per_class": 4000,
    "min_articles_per_area": 300,
    "min_abstract_words": 30,
    "split": "70/15/15 stratified by area",
    "train_rows": 54437,
    "val_rows": 14893,
    "test_rows": 14893,
    "dataset": DATASET,
    "training_gpu": TRAINING_GPU,
}

# test set, 14,893 articles (cells 53, 54, 55 and 60) / conjunto de teste, 14.893 artigos
CLASSIFIER_TEST_METRICS = {
    "test_accuracy": 0.7029,
    "test_f1_macro": 0.624,
    "test_f1_weighted": 0.702,
    "test_precision_macro": 0.628,
    "test_recall_macro": 0.632,
    "test_top3_accuracy": 0.9215,
    "test_auc_ovr_macro": 0.9706,
    "test_auc_ovr_micro": 0.9807,
    "majority_baseline_accuracy": 0.1828,
}

# retrieval check with 300 titles as queries (cell 81) / teste de recuperação com 300 títulos como consulta
RAG_EVAL_QUERIES = 300
RAG_RECALL = {
    1: {"dense": 0.747, "reranked": 0.703},
    5: {"dense": 0.970, "reranked": 0.977},
    10: {"dense": 0.990, "reranked": 0.997},
}


def rag_metrics():
    metrics = {"eval_queries": RAG_EVAL_QUERIES}
    for k, recall in RAG_RECALL.items():
        metrics[f"recall_at_{k}_dense"] = recall["dense"]
        metrics[f"recall_at_{k}_reranked"] = recall["reranked"]
    return metrics


def rag_params(rag_config):
    """Pipeline parameters from the saved rag_config.json / parâmetros do pipeline a partir do rag_config.json salvo."""
    return {
        "embedder": rag_config["embedder_name"],
        "reranker": rag_config["reranker_name"],
        "generator_notebook": rag_config["llm_name"],
        "query_prefix": rag_config["query_prefix"],
        "document_text": "abstract",
        "pooling": "cls + l2 normalize",
        "similarity": "dot product",
        "query_max_length": 128,
        "document_max_length": 512,
        "rerank_max_length": 512,
        "n_retrieve": rag_config["n_retrieve"],
        "n_context": rag_config["n_context"],
        "max_new_tokens": rag_config["max_new_tokens"],
        "do_sample": False,
        "corpus_rows": rag_config["articles_count"],
        "embedding_dim": 1024,
        "embedding_dtype": "float16",
        "dataset": DATASET,
    }
