"""bioRxiv RAG - Streamlit app.

App Streamlit: perguntas respondidas com fontes citadas e classificação da área de pesquisa.

    streamlit run app/streamlit_app.py
"""

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

# make src/ importable when the app runs outside Docker / tornar src/ importável quando o app roda fora do Docker
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biorxiv_rag import notebook_results
from biorxiv_rag.config import Settings
from biorxiv_rag.llm import GenerationError
from biorxiv_rag.serving import load_classifier, load_rag
from biorxiv_rag.tracking import QueryTracer

ALL_AREAS = "All areas"
# reranking is the slow step without a GPU, so fewer abstracts are reranked by default / a reordenação é a etapa lenta sem GPU, então menos resumos são reordenados por padrão
CPU_N_RETRIEVE = 40

EXAMPLE_QUESTIONS = [
    "What brain regions are involved in spatial navigation?",
    "How does antibiotic resistance spread between bacteria?",
    "How does SARS-CoV-2 enter human cells?",
    "Which factors drive the spread of infectious diseases during an epidemic?",
]

EXAMPLE_ARTICLE = {
    "title": "Genome assembly of a multidrug-resistant bacterial isolate",
    "abstract": (
        "We assembled the genome of a novel bacterial isolate and identified antibiotic resistance genes "
        "located on a conjugative plasmid."
    ),
}

st.set_page_config(page_title="bioRxiv RAG", page_icon="🧬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_settings():
    return Settings.from_env()


@st.cache_resource(show_spinner="Loading the retriever, the reranker and the bioRxiv abstracts (about a minute on first start)...")
def get_rag():
    return load_rag(get_settings())


@st.cache_resource(show_spinner="Loading the RoBERTa classifier...")
def get_classifier():
    return load_classifier(get_settings())


@st.cache_resource(show_spinner=False)
def get_tracer():
    return QueryTracer(get_settings())


def rag_or_stop():
    try:
        return get_rag()
    except Exception as error:
        st.error(f"The RAG pipeline could not be loaded.\n\n`{type(error).__name__}: {error}`")
        st.stop()


def retrieval_only_note(rag):
    if rag.generator_error:
        return f"Showing sources only: the answer generator is not available ({rag.generator_error})."
    return (
        "Showing sources only: no answer generator is configured. "
        "Set `HF_TOKEN` (Hugging Face Inference Providers) or `LLM_BASE_URL` to get written answers."
    )


def search_options(rag):
    """Sidebar controls for one search / controles da barra lateral para uma busca."""
    settings = get_settings()
    pipeline = rag.model
    on_gpu = pipeline.device.type == "cuda"
    default_retrieve = settings.n_retrieve or (pipeline.default_n_retrieve if on_gpu else CPU_N_RETRIEVE)
    with st.sidebar:
        st.subheader("Search settings")
        area = st.selectbox("Research area", [ALL_AREAS, *pipeline.areas], help="Search only abstracts of one bioRxiv area.")
        n_context = st.slider(
            "Sources given to the model", 1, 10, settings.n_context or pipeline.default_n_context,
            help="Best abstracts after reranking. The notebook used 6.",
        )
        n_retrieve = st.slider(
            "Abstracts reranked", 10, 200, default_retrieve, step=10,
            help="Candidates from dense retrieval that the cross-encoder rescores. The notebook used 100; "
            "fewer is faster on CPU.",
        )
        dedupe = st.toggle(
            "One version per preprint", True,
            help="The corpus holds several versions of the same preprint. Keep only the best-ranked one.",
        )
        max_new_tokens = st.slider(
            "Answer length (tokens)", 100, 1000, settings.max_new_tokens or int(pipeline.rag_config["max_new_tokens"]), step=50
        )
    return {
        "area": None if area == ALL_AREAS else area,
        "n_context": n_context,
        "n_retrieve": n_retrieve,
        "dedupe": dedupe,
        "max_new_tokens": max_new_tokens,
    }


def deployment_status(rag):
    settings = get_settings()
    tracer = get_tracer()
    pipeline = rag.model
    with st.sidebar:
        st.divider()
        st.subheader("Deployment")
        st.caption(
            f"**Models:** {rag.detail} ({rag.source})  \n"
            f"**Device:** {pipeline.device.type}  \n"
            f"**Generator:** {rag.generator.label if rag.generator else 'none (sources only)'}  \n"
            f"**Corpus:** {len(pipeline.articles_df):,} abstracts  \n"
            f"**MLflow tracing:** {'on' if tracer.enabled else 'off'}"
        )
        if settings.mlflow_tracking_uri:
            st.link_button("Open MLflow", settings.mlflow_public_url, icon=":material/monitoring:")


def render_sources(sources):
    for source in sources:
        with st.expander(f"[{source['number']}] {source['title']}"):
            link = f" · [doi:{source['doi']}](https://doi.org/{source['doi']})" if source["doi"] else ""
            st.caption(f"{source['area'] or 'No area'} · rerank score {source['rerank_score']:.2f}{link}")
            st.write(source["abstract"])


def send_feedback(trace_id, key):
    value = st.session_state.get(key)
    if value is not None:
        get_tracer().log_feedback(trace_id, helpful=bool(value))


def render_turn(index, turn):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["answer"]:
            st.markdown(turn["answer"])
        if turn["note"]:
            st.info(turn["note"])
        render_sources(turn["sources"])
        if turn["caption"]:
            st.caption(turn["caption"])
        if turn["trace_id"]:
            key = f"feedback_{index}"
            st.feedback("thumbs", key=key, on_change=send_feedback, args=(turn["trace_id"], key))


def answer_question(question, rag, options):
    """Search, stream the answer and return the finished turn / buscar, transmitir a resposta e devolver o turno pronto."""
    pipeline = rag.model
    trace = get_tracer().start(question, **options)
    turn = {"question": question, "answer": "", "note": "", "sources": [], "caption": "", "trace_id": trace.trace_id}

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Searching the abstracts...") as status:
            try:
                result = pipeline.search(
                    question,
                    n_retrieve=options["n_retrieve"],
                    n_context=options["n_context"],
                    area=options["area"],
                    dedupe=options["dedupe"],
                )
            except ValueError as error:
                status.update(label="Search failed", state="error")
                turn["note"] = str(error)
                trace.finish("", [], str(error))
                return turn
            timings = result.timings
            status.update(
                label=f"{len(result.sources)} sources from {result.n_candidates} reranked abstracts "
                f"(retrieval {timings['retrieve_s']:.1f}s, reranking {timings['rerank_s']:.1f}s)",
                state="complete",
            )
        trace.log_search(result)
        turn["sources"] = [asdict(source) for source in result.sources]
        caption = [f"retrieval {timings['retrieve_s']:.1f}s", f"reranking {timings['rerank_s']:.1f}s"]

        if rag.generator is None:
            turn["note"] = retrieval_only_note(rag)
        else:
            messages = pipeline.build_messages(question, result.sources)
            generation = rag.generator.start(messages, options["max_new_tokens"])
            trace.start_generation(messages, rag.generator.label, options["max_new_tokens"])
            failure = ""
            try:
                st.write_stream(iter(generation))
            except GenerationError as error:
                failure = str(error)
                turn["note"] = f"The answer could not be generated, so only the sources are shown. {failure}"
            trace.end_generation(generation.text, generation.model, failure)
            turn["answer"] = generation.text.strip()
            if generation.model:
                caption.insert(0, f"answered by {generation.model}")

        turn["caption"] = " · ".join(caption)
    trace.finish(turn["answer"], result.sources, turn["note"])
    return turn


def ask_page():
    st.title("🧬 Ask the bioRxiv literature")
    st.caption(
        "Retrieval-augmented answers over 99,540 bioRxiv preprint abstracts. Every answer cites the abstracts "
        "it was built from. Ask in English: the retriever was trained on English text."
    )
    rag = rag_or_stop()
    options = search_options(rag)
    deployment_status(rag)

    history = st.session_state.setdefault("history", [])
    for index, turn in enumerate(history):
        render_turn(index, turn)

    if not history:
        st.write("Try one of these:")
        for column, example in zip(st.columns(2) + st.columns(2), EXAMPLE_QUESTIONS):
            if column.button(example, width="stretch"):
                st.session_state["pending_question"] = example
                st.rerun()

    question = st.chat_input("Ask a question about the bioRxiv literature") or st.session_state.pop("pending_question", None)
    if question and question.strip():
        history.append(answer_question(question.strip(), rag, options))
        # redraw from the history so the new turn gets its feedback buttons / redesenhar do histórico para o novo turno ganhar os botões de feedback
        st.rerun()

    if history and st.sidebar.button("Clear conversation", icon=":material/delete:"):
        st.session_state["history"] = []
        st.rerun()


def fill_example_article():
    st.session_state["article_title"] = EXAMPLE_ARTICLE["title"]
    st.session_state["article_abstract"] = EXAMPLE_ARTICLE["abstract"]


def classify_page():
    st.title("🏷️ Classify an article")
    st.caption(
        "RoBERTa-large fine-tuned on bioRxiv preprints predicts the research area (25 classes) from the title "
        "and the abstract."
    )
    title = st.text_input("Title", key="article_title")
    abstract = st.text_area("Abstract", key="article_abstract", height=220)
    left, right, _ = st.columns([1, 1, 4])
    classify = left.button("Classify", type="primary", width="stretch")
    right.button("Load an example", on_click=fill_example_article, width="stretch")

    if classify:
        if not title.strip() or not abstract.strip():
            st.warning("Fill in both the title and the abstract.")
            return
        try:
            classifier = get_classifier().model
        except Exception as error:
            st.error(f"The classifier could not be loaded.\n\n`{type(error).__name__}: {error}`")
            return
        with st.spinner("Classifying..."):
            probabilities = classifier.predict_proba([title.strip()], [abstract.strip()])[0]
        ranking = (
            pd.DataFrame({"Research area": classifier.area_names, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .head(5)
        )
        st.subheader(ranking.iloc[0]["Research area"])
        st.dataframe(
            ranking,
            hide_index=True,
            width="stretch",
            column_config={
                "Probability": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0)
            },
        )
        metrics = notebook_results.CLASSIFIER_TEST_METRICS
        st.caption(
            f"On the notebook's test set the first guess was right for {metrics['test_accuracy']:.0%} of the articles "
            f"and the right area was in the top 3 for {metrics['test_top3_accuracy']:.0%}."
        )


def about_page():
    st.title("ℹ️ About this deployment")
    rag = rag_or_stop()
    deployment_status(rag)
    config = rag.model.rag_config

    st.markdown(
        f"""
**How a question is answered**

1. The question is embedded with `{config['embedder_name']}` and compared with the precomputed embeddings of
   every abstract (dense retrieval, dot product on normalized vectors).
2. The best candidates are rescored by the cross-encoder `{config['reranker_name']}`.
3. The top sources go into a grounded prompt and the language model writes an answer that cites them as [1], [2], ...

The retriever, the reranker, the embeddings and the classifier are the files saved by
`notebook/Transformer_RoBERta.ipynb`, registered in the MLflow Model Registry and loaded from there.
The notebook generated answers with `{config['llm_name']}` on a Colab GPU; here the generator is
**{rag.generator.label if rag.generator else 'not configured (sources only)'}**.
"""
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Retrieval check from the notebook")
        st.caption(f"{notebook_results.RAG_EVAL_QUERIES} article titles used as queries; a hit is the article's own abstract.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"k": k, "Recall@k (dense)": recall["dense"], "Recall@k (reranked)": recall["reranked"]}
                    for k, recall in notebook_results.RAG_RECALL.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Classifier on the notebook's test set")
        st.caption("14,893 articles, 25 research areas.")
        metrics = notebook_results.CLASSIFIER_TEST_METRICS
        first, second, third = st.columns(3)
        first.metric("Accuracy", f"{metrics['test_accuracy']:.3f}")
        second.metric("Macro F1", f"{metrics['test_f1_macro']:.3f}")
        third.metric("Top-3 accuracy", f"{metrics['test_top3_accuracy']:.3f}")
        st.caption(f"Majority-class baseline accuracy: {metrics['majority_baseline_accuracy']:.3f}.")

    st.subheader("Pipeline configuration")
    shown = {key: value for key, value in config.items() if not key.endswith("_path")}
    st.dataframe(
        pd.DataFrame({"Setting": list(shown), "Value": [str(value) for value in shown.values()]}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Answers are generated from preprint abstracts that were not peer reviewed. "
        "This is a research demo, not medical advice."
    )


st.navigation(
    [
        st.Page(ask_page, title="Ask the literature", icon="💬", default=True),
        st.Page(classify_page, title="Classify an article", icon="🏷️", url_path="classify"),
        st.Page(about_page, title="About", icon="ℹ️", url_path="about"),
    ]
).run()
