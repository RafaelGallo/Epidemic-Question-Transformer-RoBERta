"""bioRxiv RAG - retrieval, reranking and grounded generation.

Recupera os resumos mais próximos, reordena e gera respostas com fontes citadas.
"""

import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from biorxiv_rag.config import ArtifactPaths, resolve_device
from biorxiv_rag.corpus import clean_abstract, load_articles, load_embeddings, load_rag_config

QUERY_MAX_LENGTH = 128
RERANK_MAX_LENGTH = 512


@dataclass
class Source:
    number: int
    article_id: str
    title: str
    abstract: str
    area: str
    doi: str
    retrieval_score: float
    rerank_score: float

    @property
    def url(self):
        return f"https://doi.org/{self.doi}" if self.doi else ""


@dataclass
class SearchResult:
    question: str
    sources: list
    area: str = ""
    n_candidates: int = 0
    timings: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "question": self.question,
            "area": self.area,
            "n_candidates": self.n_candidates,
            "timings": self.timings,
            "sources": [asdict(source) for source in self.sources],
        }


class RagPipeline:
    def __init__(
        self,
        articles_df,
        embeddings,
        embed_tokenizer,
        embed_model,
        rerank_tokenizer,
        rerank_model,
        rag_config,
        device="cpu",
        rerank_batch_size=16,
    ):
        self.device = torch.device(device)
        # half precision only on the GPU, as in the notebook / meia precisão apenas na GPU, como no notebook
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.articles_df = articles_df
        self.embeddings = embeddings.to(self.device, self.dtype)
        self.embed_tokenizer = embed_tokenizer
        self.embed_model = embed_model.to(self.device, self.dtype).eval()
        self.rerank_tokenizer = rerank_tokenizer
        self.rerank_model = rerank_model.to(self.device, self.dtype).eval()
        self.rag_config = rag_config
        self.rerank_batch_size = rerank_batch_size
        self.areas = sorted(area for area in articles_df["area"].cat.categories if area)
        self._area_values = articles_df["area"].astype(str).to_numpy()
        # one request at a time keeps CPU threads and GPU memory under control / uma requisição por vez mantém threads de CPU e memória da GPU sob controle
        self._lock = threading.Lock()

    @classmethod
    def from_paths(cls, paths, device="auto", rerank_batch_size=16):
        """Rebuild the pipeline from the files saved by the notebook / reconstruir o pipeline a partir dos arquivos salvos pelo notebook."""
        device = resolve_device(device)
        rag_config = load_rag_config(paths.rag_config_path)
        articles_df = load_articles(paths.articles_path, rag_config)
        embeddings = load_embeddings(paths.embeddings_path, articles_df, rag_config["embedder_name"])

        # weights were saved in float16; load them in the dtype of the target device / os pesos foram salvos em float16; carregar no dtype do dispositivo de destino
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        embed_tokenizer = AutoTokenizer.from_pretrained(str(paths.embedder_dir))
        embed_model = AutoModel.from_pretrained(str(paths.embedder_dir), dtype=dtype)
        rerank_tokenizer = AutoTokenizer.from_pretrained(str(paths.reranker_dir))
        rerank_model = AutoModelForSequenceClassification.from_pretrained(str(paths.reranker_dir), dtype=dtype)
        return cls(
            articles_df,
            embeddings,
            embed_tokenizer,
            embed_model,
            rerank_tokenizer,
            rerank_model,
            rag_config,
            device=device,
            rerank_batch_size=rerank_batch_size,
        )

    @classmethod
    def from_dir(cls, artifacts_dir, device="auto", rerank_batch_size=16):
        return cls.from_paths(ArtifactPaths.from_dir(artifacts_dir), device=device, rerank_batch_size=rerank_batch_size)

    @property
    def default_n_retrieve(self):
        return int(self.rag_config["n_retrieve"])

    @property
    def default_n_context(self):
        return int(self.rag_config["n_context"])

    def embed_query(self, question):
        # bge instruction prefix applied only to queries / prefixo de instrução do bge aplicado apenas às consultas
        q_inputs = self.embed_tokenizer(
            self.rag_config["query_prefix"] + question,
            truncation=True,
            max_length=QUERY_MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            cls_vector = self.embed_model(**q_inputs).last_hidden_state[:, 0, :]
        return F.normalize(cls_vector.float(), dim=-1).to(self.dtype)

    def retrieve(self, question, n_retrieve=None, area=None):
        """Dense retrieval over every abstract / busca densa em todos os resumos."""
        n_retrieve = n_retrieve or self.default_n_retrieve
        q_vector = self.embed_query(question)
        with torch.inference_mode():
            similarities = (self.embeddings @ q_vector.T).squeeze(-1)
            limit = len(self.articles_df)
            if area:
                area_mask = torch.from_numpy(self._area_values == area).to(self.device)
                limit = int(area_mask.sum())
                if limit == 0:
                    raise ValueError(f"No articles found for area '{area}'.")
                similarities = similarities.masked_fill(~area_mask, -1e4)
            top = torch.topk(similarities, k=min(n_retrieve, limit))
        return top.indices.cpu().numpy(), top.values.float().cpu().numpy()

    def rerank(self, question, abstracts):
        """Cross-encoder scores, one per abstract / scores do cross-encoder, um por resumo."""
        scores = np.zeros(len(abstracts), dtype=np.float32)
        # sort by length to reduce padding / ordenar por tamanho para reduzir padding
        order = np.argsort([len(text) for text in abstracts])
        for start in range(0, len(order), self.rerank_batch_size):
            batch_idx = order[start:start + self.rerank_batch_size]
            r_inputs = self.rerank_tokenizer(
                [question] * len(batch_idx),
                [abstracts[i] for i in batch_idx],
                truncation=True,
                max_length=RERANK_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                scores[batch_idx] = self.rerank_model(**r_inputs).logits.squeeze(-1).float().cpu().numpy()
        return scores

    def search(self, question, n_retrieve=None, n_context=None, area=None, dedupe=True):
        """Retrieve, rerank and keep the best sources / recuperar, reordenar e manter as melhores fontes."""
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        n_retrieve = n_retrieve or self.default_n_retrieve
        n_context = n_context or self.default_n_context
        timings = {}

        with self._lock:
            timings["started_at_ns"] = time.time_ns()
            started = time.perf_counter()
            # the corpus holds several versions of the same preprint, so fetch extra rows before dropping repeats / o corpus tem várias versões do mesmo preprint, então busca linhas extras antes de remover repetidas
            top_idx, top_scores = self.retrieve(question, n_retrieve * 3 if dedupe else n_retrieve, area)
            if dedupe:
                keep = self._first_of_each_article(top_idx)[:n_retrieve]
                top_idx, top_scores = top_idx[keep], top_scores[keep]
            timings["retrieve_s"] = time.perf_counter() - started

            started = time.perf_counter()
            candidates = self.articles_df.iloc[top_idx]
            rerank_scores = self.rerank(question, candidates["abstract"].tolist())
            timings["rerank_s"] = time.perf_counter() - started

        best = np.argsort(rerank_scores)[::-1][:n_context]
        sources = [
            Source(
                number=number,
                article_id=str(candidates.iloc[i]["article_id"]),
                title=candidates.iloc[i]["title"],
                abstract=clean_abstract(candidates.iloc[i]["abstract"]),
                area=str(candidates.iloc[i]["area"]),
                doi=candidates.iloc[i]["doi"],
                retrieval_score=float(top_scores[i]),
                rerank_score=float(rerank_scores[i]),
            )
            for number, i in enumerate(best, start=1)
        ]
        return SearchResult(question, sources, area or "", len(top_idx), timings)

    def _first_of_each_article(self, row_idx):
        """Positions of the best-ranked version of each preprint / posições da versão mais bem colocada de cada preprint."""
        seen = set()
        keep = []
        rows = self.articles_df.iloc[row_idx]
        for position, (doi, title) in enumerate(zip(rows["doi"], rows["title"])):
            key = doi or title.lower()
            if key not in seen:
                seen.add(key)
                keep.append(position)
        return np.array(keep, dtype=int)

    def build_messages(self, question, sources):
        """Grounded prompt with numbered sources and titles / prompt ancorado com fontes numeradas e títulos."""
        context_block = "\n\n".join(
            f"[{source.number}] Title: {source.title}\nAbstract: {source.abstract}" for source in sources
        )
        return [
            {"role": "system", "content": self.rag_config["system_prompt"]},
            {"role": "user", "content": f"Sources:\n{context_block}\n\nQuestion: {question}"},
        ]
