"""MLflow tracing of the questions answered by the app: one trace per question.

Rastreamento no MLflow das perguntas respondidas pelo app: um trace por pergunta.
O rastreamento nunca pode derrubar uma resposta, então toda falha aqui vira apenas um aviso no log.
"""

import logging

SERVING_EXPERIMENT = "biorxiv-rag-serving"

logger = logging.getLogger(__name__)


def _quiet(method):
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            logger.warning("MLflow tracing skipped (%s): %s", method.__name__, error)
            return None

    return wrapper


class QueryTracer:
    """Opens traces in the serving experiment / abre traces no experimento de serving."""

    def __init__(self, settings):
        self.experiment_id = None
        self.error = ""
        if not settings.trace_queries or not settings.mlflow_tracking_uri:
            return
        try:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = MlflowClient()
            experiment = client.get_experiment_by_name(SERVING_EXPERIMENT)
            self.experiment_id = experiment.experiment_id if experiment else client.create_experiment(SERVING_EXPERIMENT)
        except Exception as error:
            self.error = str(error)

    @property
    def enabled(self):
        return self.experiment_id is not None

    def start(self, question, **params):
        return QueryTrace(self, question, params)

    @_quiet
    def log_feedback(self, trace_id, helpful):
        """Thumbs up or down from the user / joinha para cima ou para baixo do usuário."""
        import mlflow
        from mlflow.entities import AssessmentSource

        mlflow.log_feedback(
            trace_id=trace_id,
            name="helpful",
            value=bool(helpful),
            source=AssessmentSource(source_type="HUMAN", source_id="streamlit-user"),
        )


class QueryTrace:
    """Root span plus retrieval, rerank and generation spans / span raiz e spans de busca, reordenação e geração."""

    def __init__(self, tracer, question, params):
        self.tracer = tracer
        self.root = None
        self.generation = None
        if tracer.enabled:
            self._open(question, params)

    @property
    def trace_id(self):
        return self.root.trace_id if self.root is not None else ""

    def _span(self, name, span_type, inputs, start_time_ns=None):
        import mlflow

        # MLflow accepts the experiment only on the root span / o MLflow aceita o experimento apenas no span raiz
        return mlflow.start_span_no_context(
            name,
            span_type=span_type,
            parent_span=self.root,
            inputs=inputs,
            experiment_id=self.tracer.experiment_id if self.root is None else None,
            start_time_ns=start_time_ns,
        )

    @_quiet
    def _open(self, question, params):
        from mlflow.entities import SpanType

        self.root = self._span("rag_query", SpanType.CHAIN, {"question": question, **params})

    @_quiet
    def log_search(self, result):
        """Retrieval and rerank spans, placed on the timeline from the measured durations.

        Spans de busca e reordenação, posicionados na linha do tempo pelas durações medidas.
        """
        if self.root is None:
            return
        from mlflow.entities import SpanType

        started = result.timings["started_at_ns"]
        retrieved = started + int(result.timings["retrieve_s"] * 1e9)
        reranked = retrieved + int(result.timings["rerank_s"] * 1e9)
        span = self._span(
            "dense_retrieval", SpanType.RETRIEVER, {"question": result.question, "area": result.area}, started
        )
        span.end(outputs={"candidates": result.n_candidates}, end_time_ns=retrieved)
        span = self._span(
            "rerank", SpanType.RERANKER, {"question": result.question, "candidates": result.n_candidates}, retrieved
        )
        span.end(
            outputs=[
                {
                    "page_content": source.abstract,
                    "metadata": {
                        "doc_uri": source.url,
                        "title": source.title,
                        "area": source.area,
                        "rerank_score": source.rerank_score,
                    },
                }
                for source in result.sources
            ],
            end_time_ns=reranked,
        )

    @_quiet
    def start_generation(self, messages, model, max_new_tokens):
        if self.root is None:
            return
        from mlflow.entities import SpanType

        self.generation = self._span(
            "generate", SpanType.CHAT_MODEL, {"messages": messages, "model": model, "max_tokens": max_new_tokens}
        )

    @_quiet
    def end_generation(self, text, model, error=""):
        if self.generation is None:
            return
        self.generation.end(
            outputs=error or text, attributes={"model": model}, status="ERROR" if error else "OK"
        )
        self.generation = None

    @_quiet
    def finish(self, answer, sources, note=""):
        if self.root is None:
            return
        self.root.end(
            outputs={"answer": answer, "sources": [source.url or source.title for source in sources], "note": note},
            status="OK" if sources else "ERROR",
        )
