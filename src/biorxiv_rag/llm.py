"""bioRxiv RAG - answer generators.

Geradores de resposta: Hugging Face Inference Providers, servidor compatível com OpenAI ou modelo local.
"""

import threading

# tried in order when the first model has no live provider / tentados em ordem quando o primeiro modelo não tem provedor ativo
API_FALLBACK_MODELS = ("Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BACKENDS = ("auto", "hf_api", "openai", "local", "none")
CONNECTION_ATTEMPTS = 3

_HTTP_HINTS = {
    401: "the token was rejected (invalid, or a fine-grained token without the 'Make calls to Inference Providers' permission)",
    402: "the Hugging Face account has no Inference Providers credits left",
    403: "the token is not allowed to call this model",
    404: "the model is not served by this endpoint",
    429: "rate limit reached",
}


class GenerationError(RuntimeError):
    """The generator failed / o gerador falhou."""


class Generation:
    """One answer being generated: iterate for the text pieces, then read .model and .text.

    Uma resposta sendo gerada: itere para receber os trechos e depois leia .model e .text.
    """

    def __init__(self, pieces):
        self._pieces = pieces
        self.model = ""
        self.text = ""

    def __iter__(self):
        for model, piece in self._pieces:
            self.model = model
            self.text += piece
            yield piece


def _status(error):
    return getattr(getattr(error, "response", None), "status_code", None)


def _describe(error):
    status = _status(error)
    if status in _HTTP_HINTS:
        return f"HTTP {status}: {_HTTP_HINTS[status]}"
    # the server message often sits on a later line, after the request id / a mensagem do servidor costuma vir numa linha seguinte, depois do id da requisição
    message = " ".join(str(error).split()) or type(error).__name__
    return f"HTTP {status}: {message[:300]}" if status else message[:300]


class ChatApiGenerator:
    """Chat completions over HTTP, streamed piece by piece / chat completions via HTTP, com streaming."""

    def __init__(self, models, backend, token="", provider="auto", base_url=""):
        from huggingface_hub import InferenceClient

        self.models = list(models)
        self.backend = backend
        if base_url:
            # any OpenAI-compatible server: Ollama, vLLM, LM Studio / qualquer servidor compatível com OpenAI
            self.client = InferenceClient(base_url=base_url, api_key=token or "not-needed", timeout=300)
        else:
            self.client = InferenceClient(provider=provider or "auto", api_key=token, timeout=300)

    @property
    def label(self):
        return f"{self.models[0]} ({self.backend})"

    def start(self, messages, max_new_tokens):
        return Generation(self._pieces(messages, max_new_tokens))

    def _pieces(self, messages, max_new_tokens):
        failures = []
        for model in self.models:
            for attempt in range(CONNECTION_ATTEMPTS):
                produced = False
                try:
                    # temperature 0 mirrors do_sample=False of the notebook / temperatura 0 reproduz o do_sample=False do notebook
                    chunks = self.client.chat_completion(
                        messages=messages,
                        model=model,
                        max_tokens=max_new_tokens,
                        temperature=0.0,
                        stream=True,
                    )
                    for chunk in chunks:
                        if chunk.choices and chunk.choices[0].delta.content:
                            produced = True
                            yield model, chunk.choices[0].delta.content
                except Exception as error:
                    if produced:
                        raise GenerationError(f"{model} stopped while answering: {_describe(error)}") from error
                    # no HTTP status means the request never reached the server, so it is worth another try
                    # sem status HTTP a requisição nem chegou ao servidor, então vale tentar de novo
                    if _status(error) is None and attempt + 1 < CONNECTION_ATTEMPTS:
                        continue
                    failures.append(f"{model}: {_describe(error)}")
                    break
                if produced:
                    return
                failures.append(f"{model}: empty answer")
                break
        raise GenerationError(" | ".join(failures))


class LocalGenerator:
    """Causal LM loaded with transformers, as in the notebook / LM causal carregado com transformers, como no notebook."""

    backend = "local"

    def __init__(self, model, device="cpu", token=""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token or None)
        self.llm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, token=token or None).to(device).eval()
        self._lock = threading.Lock()

    @property
    def label(self):
        return f"{self.model} (local, {self.llm.device.type})"

    def start(self, messages, max_new_tokens):
        return Generation(self._pieces(messages, max_new_tokens))

    def _pieces(self, messages, max_new_tokens):
        from transformers import TextIteratorStreamer

        # chat template returns a dict in transformers 5.x / o chat template retorna um dicionário no transformers 5.x
        chat_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        failures = []

        def run():
            try:
                self.llm.generate(
                    **chat_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )
            except Exception as error:
                failures.append(error)
                # unblock the reader below / desbloquear o leitor abaixo
                streamer.text_queue.put(streamer.stop_signal)

        with self._lock:
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            for piece in streamer:
                if piece:
                    yield self.model, piece
            worker.join()
        if failures:
            raise GenerationError(f"{self.label}: {_describe(failures[0])}") from failures[0]


def build_generator(settings, notebook_model, device="cpu"):
    """Pick the generator from the settings; None means retrieval only.

    Escolhe o gerador pelas configurações; None significa apenas recuperação, sem geração.
    """
    backend = settings.llm_backend
    if backend not in BACKENDS:
        raise ValueError(f"LLM_BACKEND must be one of {BACKENDS}, got '{backend}'.")

    if backend == "auto":
        if settings.llm_base_url:
            backend = "openai"
        elif settings.hf_token:
            backend = "hf_api"
        else:
            backend = "none"

    if backend == "none":
        return None
    if backend == "openai":
        if not settings.llm_base_url or not settings.llm_model:
            raise ValueError("LLM_BACKEND=openai needs LLM_BASE_URL and LLM_MODEL.")
        return ChatApiGenerator(
            [settings.llm_model], backend, token=settings.llm_api_key, base_url=settings.llm_base_url
        )
    if backend == "hf_api":
        if not settings.hf_token:
            raise ValueError("LLM_BACKEND=hf_api needs HF_TOKEN.")
        # the notebook model first, then larger and smaller siblings / primeiro o modelo do notebook, depois os irmãos maior e menor
        models = [settings.llm_model] if settings.llm_model else [notebook_model, *API_FALLBACK_MODELS]
        return ChatApiGenerator(models, backend, token=settings.hf_token, provider=settings.llm_provider)
    return LocalGenerator(settings.llm_model or DEFAULT_LOCAL_MODEL, device=device, token=settings.hf_token)


def try_build_generator(settings, pipeline):
    """Generator and error message; a broken generator must not take retrieval down with it.

    Gerador e mensagem de erro; um gerador quebrado não pode derrubar a recuperação.
    """
    try:
        return build_generator(settings, pipeline.rag_config["llm_name"], device=pipeline.device.type), ""
    except Exception as error:
        return None, str(error)
