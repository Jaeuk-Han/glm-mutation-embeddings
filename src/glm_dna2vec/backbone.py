from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .utils import l2_normalize


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class DNA2VecBackbone(torch.nn.Module):
    """Backbone encoder wrapper (default: roychowdhuryresearch/dna2vec).

    - tokenize(seqs) -> TokenizedBatch
    - forward_embeds(input_ids, attention_mask) -> [B, H] (L2-normalized)
    """

    def __init__(self, model_id: str = "roychowdhuryresearch/dna2vec", max_length: int = 512):
        super().__init__()
        self.model_id = model_id
        self.max_length = int(max_length)

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.config = cfg

        self.model = AutoModel.from_pretrained(
            model_id,
            config=cfg,
            trust_remote_code=True,
        )

        # Tokenizer: dna2vec sometimes needs fallback to the original tokenizer id
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                try:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                except Exception:
                    pass

        hidden = getattr(cfg, "hidden_size", None)
        if hidden is None:
            # dna2vec config should have hidden_size, but keep a small fallback.
            for attr in ["d_model", "embed_dim", "dim", "hidden_dim"]:
                if hasattr(cfg, attr):
                    hidden = getattr(cfg, attr)
                    if hidden is not None:
                        break

        if hidden is None:
            # Dummy forward to infer
            with torch.no_grad():
                dummy_ids = torch.zeros(1, 8, dtype=torch.long)
                out = self.model(input_ids=dummy_ids)
                last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                hidden = last_hidden.shape[-1]

        self.hidden_size = int(hidden)

    def tokenize(self, seqs: List[str], max_length: Optional[int] = None) -> TokenizedBatch:
        ml = self.max_length if max_length is None else int(max_length)
        enc: Dict[str, torch.Tensor] = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=ml,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            if self.tokenizer.pad_token_id is not None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)
        return TokenizedBatch(input_ids=input_ids, attention_mask=attention_mask)

    def forward_embeds(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return pooled backbone embeddings as [B, H] (L2-normalized)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # dna2vec accepts attention_mask; keep output_hidden_states off for speed.
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last_hidden = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            last_hidden = out[0]
        elif isinstance(out, torch.Tensor):
            last_hidden = out
        else:
            raise TypeError(f"Unexpected model output type: {type(out)}")

        # Mean pool over sequence length
        if last_hidden.dim() == 3:
            h = last_hidden.mean(dim=1)
        elif last_hidden.dim() == 2:
            h = last_hidden
        else:
            raise ValueError(f"Unexpected hidden shape: {tuple(last_hidden.shape)}")

        return l2_normalize(h)
