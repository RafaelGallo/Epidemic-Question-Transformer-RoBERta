"""Corpus loading must stay aligned with the saved embeddings / o corpus precisa continuar alinhado aos embeddings salvos."""

import pandas as pd
import pytest
import torch

from biorxiv_rag.corpus import CorpusMismatchError, clean_abstract, load_articles, load_embeddings, normalize_areas

LONG_ABSTRACT = " ".join(["word"] * 40)


def write_articles(path):
    pd.DataFrame(
        {
            "article_id": ["001", "002", "003", "004"],
            "title": ["First", "Second", "", "Fourth"],
            "abstract": [LONG_ABSTRACT, "too short", LONG_ABSTRACT, "Abstract " + LONG_ABSTRACT],
            "subject": ["NEUROSCIENCE", "Genomics", "Genomics", "Neuroscience"],
            "doi": ["10.1101/001", "10.1101/002", "10.1101/003", "10.1101/004"],
        }
    ).to_parquet(path, index=False)


def test_clean_abstract_drops_only_the_leading_label():
    assert clean_abstract("Abstract We studied cells.") == "We studied cells."
    assert clean_abstract("ABSTRACT: Background Cells divide.") == "Background Cells divide."
    assert clean_abstract("Abstract thinking is studied here.") == "Abstract thinking is studied here."
    assert clean_abstract("We studied the Abstract of cells.") == "We studied the Abstract of cells."


def test_normalize_areas_merges_spelling_variants():
    subjects = pd.Series(["Neuroscience", "NEUROSCIENCE", "Neuroscience", "Clinical_Trials", "Clinical Trials", ""])
    assert normalize_areas(subjects).tolist() == [
        "Neuroscience", "Neuroscience", "Neuroscience", "Clinical Trials", "Clinical Trials", "",
    ]


def test_load_articles_applies_the_notebook_filter(tmp_path):
    write_articles(tmp_path / "articles.parquet")
    articles_df = load_articles(tmp_path / "articles.parquet", {"articles_count": 2, "first_article_id": "001"})
    # short abstracts and empty titles are dropped / resumos curtos e títulos vazios são removidos
    assert articles_df["article_id"].tolist() == ["001", "004"]
    assert articles_df["area"].nunique() == 1
    assert articles_df.loc[1, "abstract"].startswith("Abstract ")


def test_load_articles_rejects_a_corpus_that_does_not_match(tmp_path):
    write_articles(tmp_path / "articles.parquet")
    with pytest.raises(CorpusMismatchError):
        load_articles(tmp_path / "articles.parquet", {"articles_count": 3, "first_article_id": "001"})
    with pytest.raises(CorpusMismatchError):
        load_articles(tmp_path / "articles.parquet", {"articles_count": 2, "first_article_id": "004"})


def test_load_embeddings_checks_rows_order_and_model(tmp_path):
    write_articles(tmp_path / "articles.parquet")
    articles_df = load_articles(tmp_path / "articles.parquet")
    cache = {"model": "embedder", "count": 2, "first_id": "001", "embeddings": torch.zeros((2, 4), dtype=torch.float16)}
    torch.save(cache, tmp_path / "embeddings.pt")
    assert load_embeddings(tmp_path / "embeddings.pt", articles_df, "embedder").shape == (2, 4)
    with pytest.raises(CorpusMismatchError):
        load_embeddings(tmp_path / "embeddings.pt", articles_df, "another-embedder")
    torch.save({**cache, "first_id": "004"}, tmp_path / "embeddings.pt")
    with pytest.raises(CorpusMismatchError):
        load_embeddings(tmp_path / "embeddings.pt", articles_df, "embedder")
