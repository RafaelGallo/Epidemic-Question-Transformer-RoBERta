"""Article corpus and cached abstract embeddings.

Corpus de artigos e embeddings dos resumos em cache.
"""

import json
import re

import pandas as pd
import torch

MIN_ABSTRACT_WORDS = 30

# section label that many abstracts carry at the start / rótulo de seção que muitos resumos trazem no início
_LEADING_LABEL = re.compile(r"^\s*(?:Abstract|ABSTRACT)\s*[:.]?\s+(?=[A-Z0-9\"“(\[])")


class CorpusMismatchError(RuntimeError):
    """The articles do not line up with the saved embeddings / os artigos não batem com os embeddings salvos."""


def load_rag_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_abstract(text):
    """Drop the leading 'Abstract' label for display and prompts / remover o rótulo 'Abstract' inicial para exibição e prompts."""
    return _LEADING_LABEL.sub("", text or "", count=1).strip()


def normalize_areas(subjects):
    """Merge spelling variants of the research area / unificar variações de grafia da área de pesquisa.

    'BIOINFORMATICS' and 'Bioinformatics' become one area, named after the most common spelling.
    """
    cleaned = subjects.fillna("").str.replace("_", " ").str.strip()
    keys = cleaned.str.lower()
    canonical = cleaned.groupby(keys).agg(lambda values: values.value_counts().idxmax())
    return keys.map(canonical).fillna("")


def load_articles(articles_path, rag_config=None):
    """Load the articles with the same filter used to build the embeddings.

    Carrega os artigos com o mesmo filtro usado para gerar os embeddings.
    """
    articles_df = pd.read_parquet(articles_path, columns=["article_id", "title", "abstract", "subject", "doi"])
    abstract_words = articles_df["abstract"].str.split().str.len().fillna(0)
    articles_df = articles_df[
        (abstract_words >= MIN_ABSTRACT_WORDS) & (articles_df["title"].str.len() > 0)
    ].reset_index(drop=True)

    # row order is the only link to the embedding matrix / a ordem das linhas é o único vínculo com a matriz de embeddings
    if rag_config is not None:
        if len(articles_df) != rag_config["articles_count"]:
            raise CorpusMismatchError(
                f"article count differs from the saved configuration: {len(articles_df)} != {rag_config['articles_count']}"
            )
        if str(articles_df.loc[0, "article_id"]) != str(rag_config["first_article_id"]):
            raise CorpusMismatchError("article order differs from the saved configuration")

    articles_df["area"] = normalize_areas(articles_df["subject"]).astype("category")
    return articles_df


def load_embeddings(embeddings_path, articles_df, embedder_name=None):
    """Load the cached abstract embeddings and validate them against the corpus.

    Carrega os embeddings em cache e valida contra o corpus.
    """
    cached = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    embeddings = cached["embeddings"]
    if cached["count"] != len(articles_df) or embeddings.shape[0] != len(articles_df):
        raise CorpusMismatchError(
            f"embeddings hold {embeddings.shape[0]} rows but the corpus has {len(articles_df)} articles"
        )
    if str(cached["first_id"]) != str(articles_df.loc[0, "article_id"]):
        raise CorpusMismatchError("embeddings were built for a different article order")
    if embedder_name and cached["model"] != embedder_name:
        raise CorpusMismatchError(f"embeddings were built with {cached['model']}, not {embedder_name}")
    return embeddings
