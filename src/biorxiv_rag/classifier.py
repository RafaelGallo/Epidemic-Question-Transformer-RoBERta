"""bioRxiv RoBERTa classifier - research area from title and abstract.

Classificador RoBERTa do bioRxiv: área de pesquisa a partir do título e do resumo.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from biorxiv_rag.config import resolve_device

MAX_LENGTH = 384


class AreaClassifier:
    def __init__(self, model_dir, device="auto"):
        self.device = torch.device(resolve_device(device))
        # bfloat16 on the GPU as in the notebook, float32 on the CPU / bfloat16 na GPU como no notebook, float32 na CPU
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), dtype=dtype).to(self.device).eval()
        self.area_names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_proba(self, titles, abstracts, batch_size=8):
        """Probabilities for every area, one row per article / probabilidades para cada área, uma linha por artigo."""
        probabilities = []
        for start in range(0, len(titles), batch_size):
            # tokenize title and abstract as a text pair / tokenizar título e resumo como par de textos
            inputs = self.tokenizer(
                list(titles[start:start + batch_size]),
                list(abstracts[start:start + batch_size]),
                truncation="only_second",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits.float()
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probabilities) if probabilities else np.zeros((0, len(self.area_names)), dtype=np.float32)

    def predict(self, title, abstract, top_k=3):
        """Most likely areas for one article / áreas mais prováveis para um artigo."""
        probs = self.predict_proba([title], [abstract])[0]
        return [
            {"area": self.area_names[idx], "probability": float(probs[idx])}
            for idx in np.argsort(probs)[::-1][:top_k]
        ]
