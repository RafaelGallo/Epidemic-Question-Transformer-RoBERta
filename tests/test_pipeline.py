"""Search logic with the models replaced by fixed scores / lógica de busca com os modelos trocados por scores fixos."""

import threading

import numpy as np
import pandas as pd
import pytest

from biorxiv_rag.pipeline import RagPipeline

RAG_CONFIG = {
    "n_retrieve": 4,
    "n_context": 2,
    "system_prompt": "Answer using ONLY the numbered sources provided.",
}


class FixedScorePipeline(RagPipeline):
    """Dense scores follow row order; rerank scores are given / scores densos seguem a ordem das linhas; scores do rerank são dados."""

    def __init__(self, articles_df, rerank_scores):
        self.articles_df = articles_df
        self.rag_config = RAG_CONFIG
        self.rerank_scores = rerank_scores
        self._area_values = articles_df["area"].to_numpy()
        self._lock = threading.Lock()

    def retrieve(self, question, n_retrieve=None, area=None):
        rows = np.arange(len(self.articles_df))
        if area:
            rows = rows[self._area_values == area]
            if len(rows) == 0:
                raise ValueError(f"No articles found for area '{area}'.")
        rows = rows[:n_retrieve]
        return rows, 1.0 - rows / 10

    def rerank(self, question, abstracts):
        return np.array([self.rerank_scores[text] for text in abstracts], dtype=np.float32)


@pytest.fixture
def pipeline():
    articles_df = pd.DataFrame(
        {
            "article_id": ["1", "1", "2", "3", "4"],
            "title": ["Paper A", "Paper A", "Paper B", "Paper C", "Paper D"],
            "abstract": ["Abstract A v1", "A v2", "B", "C", "D"],
            "area": ["Neuroscience", "Neuroscience", "Genomics", "Genomics", "Neuroscience"],
            # rows 0 and 1 are two versions of the same preprint / as linhas 0 e 1 são duas versões do mesmo preprint
            "doi": ["10.1101/a", "10.1101/a", "10.1101/b", "", "10.1101/d"],
        }
    )
    return FixedScorePipeline(articles_df, {"Abstract A v1": 3.0, "A v2": 2.5, "B": 1.0, "C": 2.0, "D": 0.5})


def test_search_orders_sources_by_rerank_score(pipeline):
    result = pipeline.search("question", n_retrieve=4, n_context=3)
    assert [source.title for source in result.sources] == ["Paper A", "Paper C", "Paper B"]
    assert [source.number for source in result.sources] == [1, 2, 3]
    assert result.sources[0].rerank_score == pytest.approx(3.0)
    assert result.n_candidates == 4


def test_search_keeps_one_version_per_preprint(pipeline):
    result = pipeline.search("question", n_retrieve=4, n_context=4)
    assert [source.doi for source in result.sources] == ["10.1101/a", "", "10.1101/b", "10.1101/d"]


def test_search_without_dedupe_matches_the_notebook(pipeline):
    result = pipeline.search("question", n_retrieve=4, n_context=2, dedupe=False)
    assert [source.doi for source in result.sources] == ["10.1101/a", "10.1101/a"]


def test_search_filters_by_area(pipeline):
    result = pipeline.search("question", area="Genomics")
    assert {source.area for source in result.sources} == {"Genomics"}
    with pytest.raises(ValueError):
        pipeline.search("question", area="Zoology")


def test_search_rejects_an_empty_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.search("   ")


def test_prompt_follows_the_notebook_format(pipeline):
    result = pipeline.search("What is A?", n_retrieve=4, n_context=2)
    messages = pipeline.build_messages("What is A?", result.sources)
    assert messages[0] == {"role": "system", "content": RAG_CONFIG["system_prompt"]}
    # the leading 'Abstract' label is dropped from the prompt / o rótulo 'Abstract' inicial sai do prompt
    assert messages[1]["content"] == (
        "Sources:\n[1] Title: Paper A\nAbstract: A v1\n\n[2] Title: Paper C\nAbstract: C\n\nQuestion: What is A?"
    )
