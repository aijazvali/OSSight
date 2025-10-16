# mm_modules.py  — SigLIP → ProjectionAdapter → CrossAttentionFusion → Qwen (4-bit)
from __future__ import annotations
import contextlib, re, difflib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import (
    AutoProcessor,
    SiglipVisionModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# -----------------------
# Projection Adapter
# -----------------------
class ProjectionAdapter(nn.Module):
    """Small MLP mapping vision token dim (v_dim) → LLM hidden size (H). Trainable."""
    def __init__(self, v_dim: int, h_dim: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        mid = hidden or max(v_dim, h_dim)
        self.net = nn.Sequential(
            nn.Linear(v_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, h_dim),
        )
    def forward(self, x):  # [B, N, v_dim]
        return self.net(x) # [B, N, H]


# -----------------------
# Cross-Attention Fusion (Q=text; K/V=vision)
# -----------------------
class CrossAttentionFusion(nn.Module):
    """
    Inject visual context into the text stream using cross-attention.
    - Resamples vision tokens to M for compact 'visual memory'
    - Gated residual into text stream + LayerNorm
    """
    def __init__(self, h_dim: int, max_vision_tokens: int = 64, num_heads: int = 16,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0, use_fp32_attention: bool = False):
        super().__init__()
        assert h_dim % num_heads == 0, "h_dim must be divisible by num_heads"
        self.h = h_dim
        self.M = max_vision_tokens
        self.num_heads = num_heads
        self.use_fp32_attention = use_fp32_attention

        self.resample_ln = nn.LayerNorm(h_dim)

        self.q_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.k_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.v_proj = nn.Linear(h_dim, h_dim, bias=False)
        self.o_proj = nn.Linear(h_dim, h_dim, bias=False)

        self.pre_txt_ln = nn.LayerNorm(h_dim)
        self.pre_vis_ln = nn.LayerNorm(h_dim)
        self.post_ln    = nn.LayerNorm(h_dim)
        self.gate       = nn.Parameter(torch.zeros(1))

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    @staticmethod
    def _split_heads(x, nh):
        B, T, H = x.shape
        hd = H // nh
        return x.view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]

    @staticmethod
    def _merge_heads(x):
        B, nh, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, nh * hd)

    def _resample_visual(self, v_tokens):
        # [B, N, H] -> [B, M, H] via linear interpolation over the token axis
        B, N, H = v_tokens.shape
        if N == self.M:
            return v_tokens
        x = self.resample_ln(v_tokens)
        x = x.transpose(1, 2)                 # [B, H, N]
        x = F.interpolate(x, size=self.M, mode="linear", align_corners=False)
        x = x.transpose(1, 2)                 # [B, M, H]
        return x

    def forward(self, text_embeds, visual_tokens):
        vm = self._resample_visual(visual_tokens)                 # [B, M, H]
        q = self.q_proj(self.pre_txt_ln(text_embeds))             # [B, L, H]
        k = self.k_proj(self.pre_vis_ln(vm))                      # [B, M, H]
        v = self.v_proj(self.pre_vis_ln(vm))                      # [B, M, H]

        q = self._split_heads(q, self.num_heads)                  # [B, nh, L, hd]
        k = self._split_heads(k, self.num_heads)                  # [B, nh, M, hd]
        v = self._split_heads(v, self.num_heads)                  # [B, nh, M, hd]

        ctx = (torch.autocast(device_type=text_embeds.device.type, dtype=torch.float32, enabled=True)
               if self.use_fp32_attention else contextlib.nullcontext())
        with ctx:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )                                                     # [B, nh, L, hd]

        out = self._merge_heads(out)                              # [B, L, H]
        out = self.o_proj(out)
        out = self.proj_dropout(out)

        fused = text_embeds + torch.sigmoid(self.gate) * out
        fused = self.post_ln(fused)
        return fused


# -----------------------
# Vision backbone (SigLIP), frozen, dynamic v_dim
# -----------------------
class VisionBackbone(nn.Module):
    """
    Wrap SigLIP vision model. Returns patch tokens [B, N, v_dim].
    Dynamically detects output dim (e.g., 1152 for so400m).
    """
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", dtype=torch.bfloat16):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name, dtype=dtype, device_map="auto")
        for p in self.model.parameters():
            p.requires_grad = False
        self._out_dim: Optional[int] = None

    @torch.no_grad()
    def infer_out_dim(self) -> int:
        d = getattr(self.model.config, "hidden_size", None)
        if d is None:
            vc = getattr(self.model.config, "vision_config", None)
            d = getattr(vc, "hidden_size", None) if vc is not None else None
        if d is not None:
            self._out_dim = int(d)
            return self._out_dim
        # fallback: probe
        dummy = Image.fromarray((np.ones((384,384,3))*127).astype("uint8"))
        device = next(self.model.parameters()).device
        batch = self.processor(images=[dummy], return_tensors="pt").to(device)
        out = self.model(**batch)
        self._out_dim = int(out.last_hidden_state.shape[-1])
        return self._out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim if self._out_dim is not None else self.infer_out_dim()

    @torch.no_grad()
    def forward(self, pil_list: List[Image.Image]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.processor(images=pil_list, return_tensors="pt").to(device)
        out = self.model(**batch)  # last_hidden_state [B, N, v_dim]
        if self._out_dim is None:
            self._out_dim = int(out.last_hidden_state.shape[-1])
        return out.last_hidden_state


# -----------------------
# Config
# -----------------------
@dataclass
class MMConfig:
    vision_model_name: str = "google/siglip-so400m-patch14-384"
    llm_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"      # "bfloat16" (recommended) or "float16"
    use_4bit: bool = True
    max_vision_tokens: int = 64
    num_heads: int = -1          # auto from LLM if -1
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    use_fp32_attention: bool = False


# -----------------------
# Full multimodal wrapper
# -----------------------
class VisionMiniVLM(nn.Module):
    """
    SigLIP (frozen) → ProjectionAdapter → CrossAttentionFusion → LLM via inputs_embeds
    Trainable: Adapter + Fusion
    """
    def __init__(self, cfg: MMConfig):
        super().__init__()
        self.cfg = cfg
        torch_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

        # Vision
        self.vision = VisionBackbone(cfg.vision_model_name, dtype=torch_dtype)
        v_dim = self.vision.out_dim
        print(f"[VisionMiniVLM] Detected SigLIP v_dim = {v_dim}")

        # LLM (optionally 4-bit)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if cfg.use_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name_or_path,
            dtype=torch_dtype,                 # HF >=4.45 uses 'dtype'
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        for p in self.llm.parameters():
            p.requires_grad = False

        self.h_dim = self.llm.config.hidden_size
        fuse_heads = cfg.num_heads if cfg.num_heads > 0 else getattr(self.llm.config, "num_attention_heads", 16)

        self.adapter = ProjectionAdapter(v_dim=v_dim, h_dim=self.h_dim, hidden=None, dropout=0.1)
        self.fusion  = CrossAttentionFusion(
            h_dim=self.h_dim,
            max_vision_tokens=cfg.max_vision_tokens,
            num_heads=fuse_heads,
            attn_dropout=cfg.attn_dropout,
            proj_dropout=cfg.proj_dropout,
            use_fp32_attention=cfg.use_fp32_attention,
        )
        self.embed_tokens = self.llm.get_input_embeddings()

        # Ensure trainables are on same device/dtype as embeddings
        target_device = next(self.embed_tokens.parameters()).device
        target_dtype  = next(self.embed_tokens.parameters()).dtype
        self.adapter.to(target_device, dtype=target_dtype)
        self.fusion.to(target_device, dtype=target_dtype)

        # bad-word ids to avoid hyphen spam in decode
        try:
            self._bad_word_ids = self.tokenizer(["-","•","—"], add_special_tokens=False).input_ids
        except Exception:
            self._bad_word_ids = None

    @property
    def device(self):
        return next(self.embed_tokens.parameters()).device

    @property
    def embed_dtype(self):
        return next(self.embed_tokens.parameters()).dtype

    # ---------- Tokenize helper ----------
    def tokenize(self, texts: List[str]):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)

    # ---------- Vision path ----------
    def encode_vision(self, pil_list: List[Image.Image]) -> torch.Tensor:
        v = self.vision(pil_list)                                     # [B, N, v_dim]
        v = v.to(self.device, dtype=self.embed_dtype)
        v = self.adapter(v)                                           # [B, N, H]
        return v

    # ---------- Build fused inputs_embeds ----------
    def build_inputs_embeds(self, input_ids: torch.Tensor, visual_tokens: torch.Tensor):
        t_emb = self.embed_tokens(input_ids.to(self.device)).to(self.embed_dtype)  # [B, L, H]
        vis  = visual_tokens.to(self.device, dtype=self.embed_dtype)               # [B, N, H]
        return self.fusion(t_emb, vis)                                             # [B, L, H]

    # ---------- Forward (for training) ----------
    def forward(self, images: List[Image.Image], input_ids: torch.Tensor,
                labels: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, **llm_kwargs):
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        v_tokens = self.encode_vision(images)
        inputs_embeds = self.build_inputs_embeds(input_ids, v_tokens)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(self.device) if labels is not None else None,
            **llm_kwargs,
        )
        return out

    # ---------- Chat templating ----------
    def _build_chat(self, system_msg: str, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            return f"{system_msg}\nUser: {user_msg}\nAssistant:"

    # ---------- General VQA (free-form, ChatGPT-like) ----------
    @torch.no_grad()
    def generate_freeform(self, images: List[Image.Image], question: str,
                          max_new_tokens=96, min_new_tokens=8,
                          temperature=0.5, top_p=0.9) -> str:
        self.eval()
        system = "You are a helpful vision-language assistant. Be precise and concise."
        prompt_text = self._build_chat(system, question)

        tok = self.tokenize([prompt_text])
        input_ids = tok["input_ids"].to(self.device)
        attn_mask = tok["attention_mask"].to(self.device)

        v_tokens = self.encode_vision(images)
        inputs_embeds = self.build_inputs_embeds(input_ids, v_tokens)

        out_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    # ---------- EO land-cover short answer (deterministic) ----------
    @torch.no_grad()
    def generate_landcover(self, images: List[Image.Image], question: str,
                           max_new_tokens: int = 24, min_new_tokens: int = 6) -> str:
        self.eval()
        system_msg = "You are an EO (Earth-Observation) analyst. Reply in one short sentence."
        answer_prefix = "The main land cover is "

        # Build prompt with open assistant turn that starts with our prefix
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": answer_prefix},
        ]
        try:
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        except Exception:
            prompt_text = f"{system_msg}\nUser: {question}\nAssistant: {answer_prefix}"

        tok = self.tokenize([prompt_text])
        input_ids = tok["input_ids"].to(self.device)
        attn_mask = tok["attention_mask"].to(self.device)

        v_tokens = self.encode_vision(images)
        inputs_embeds = self.build_inputs_embeds(input_ids, v_tokens)

        kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if self._bad_word_ids:
            kwargs["bad_words_ids"] = self._bad_word_ids

        out_ids = self.llm.generate(**kwargs)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if answer_prefix in text:
            text = text.split(answer_prefix, 1)[-1]
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[\.!\?].*$", "", text).strip()
        return text

    # ---------- Map free-form to class label (optional helper) ----------
    def predict_label(self, images: List[Image.Image], question: str, class_list: Optional[List[str]] = None) -> str:
        text = self.generate_landcover(images, question)
        if not class_list:
            return text
        t = text.lower()
        for c in class_list:
            name = c.replace("_"," ").lower()
            if name in t:
                return c
        best = difflib.get_close_matches(
            t, [c.replace("_"," ").lower() for c in class_list], n=1, cutoff=0.1
        )
        if best:
            idx = [c.replace("_"," ").lower() for c in class_list].index(best[0])
            return class_list[idx]
        return text
