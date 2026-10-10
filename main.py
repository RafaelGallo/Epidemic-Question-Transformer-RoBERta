"""bioRxiv RAG - ask one question from the terminal.

Faz uma pergunta pelo terminal: imprime a resposta e as fontes citadas.

    python main.py "How does antibiotic resistance spread between bacteria?"
    python main.py "What drives SARS-CoV-2 transmission?" --area Epidemiology --no-answer
"""

import argparse
import sys
from pathlib import Path

# make src/ importable without installing the package / tornar src/ importável sem instalar o pacote
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from biorxiv_rag.config import Settings
from biorxiv_rag.llm import GenerationError
from biorxiv_rag.serving import load_rag


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--area", default=None, help="search only one research area, for example Epidemiology")
    parser.add_argument("--n-retrieve", type=int, default=None, help="abstracts reranked (notebook: 100)")
    parser.add_argument("--n-context", type=int, default=None, help="sources given to the model (notebook: 6)")
    parser.add_argument("--keep-versions", action="store_true", help="keep every version of the same preprint")
    parser.add_argument("--no-answer", action="store_true", help="print the sources only")
    args = parser.parse_args()

    settings = Settings.from_env()
    rag = load_rag(settings)
    pipeline = rag.model
    result = pipeline.search(
        args.question,
        n_retrieve=args.n_retrieve or settings.n_retrieve,
        n_context=args.n_context or settings.n_context,
        area=args.area,
        dedupe=not args.keep_versions,
    )

    print("Question:", args.question)
    print()
    if rag.generator is not None and not args.no_answer:
        max_new_tokens = settings.max_new_tokens or int(pipeline.rag_config["max_new_tokens"])
        generation = rag.generator.start(pipeline.build_messages(args.question, result.sources), max_new_tokens)
        try:
            for piece in generation:
                print(piece, end="", flush=True)
            print("\n")
        except GenerationError as error:
            print(f"\n(no answer: {error})\n")
    elif not args.no_answer:
        print(f"(sources only: {rag.generator_error or 'no answer generator is configured'})\n")

    print("Sources:")
    for source in result.sources:
        print(f"[{source.number}] score={source.rerank_score:.2f} | {source.area} | {source.title[:90]} | doi: {source.doi}")


if __name__ == "__main__":
    main()
