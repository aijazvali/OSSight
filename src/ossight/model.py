from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoModel, BitsAndBytesConfig

@dataclass
class FusionConfig:
    llm_hidden: int
    proj_hidden: int = 4096
    heads: int = 8
    max_vision_tokens: int = 256
    init_gate: float = -2.0

class ProjectionAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden: int, heads: int, init_gate: float = -2.0):
        super().__init__()
        self.norm_t = nn.LayerNorm(hidden)
        self.norm_v = nn.LayerNorm(hidden)
        self.attn   = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True)
        self.proj   = nn.Linear(hidden, hidden)
        self.gate   = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))
    def forward(self, t_embed, v_embed):
        t = self.norm_t(t_embed)
        v = self.norm_v(v_embed)
        attn_out, _ = self.attn(query=t, key=v, value=v, need_weights=False)
        out = self.proj(attn_out)
        return t_embed + torch.sigmoid(self.gate) * out

@torch.no_grad()
def vision_tokens_from_pils(vision_model, vision_processor, pils, device):
    batch = vision_processor(images=pils, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    vision_model.to(device)
    # Handle common model variants
    try:
        out = vision_model.vision_model(**batch); seq = out.last_hidden_state
        return seq
    except Exception:
        try:
            out = vision_model(**batch); seq = out.last_hidden_state
            return seq
        except Exception:
            pooled = vision_model.get_image_features(**batch)
            return pooled.unsqueeze(1) if pooled.dim() == 2 else pooled

def resample_seq(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
    B, N, D = tokens.shape
    if N == max_tokens:
        return tokens
    x = tokens.transpose(1, 2)
    x = F.interpolate(x, size=max_tokens, mode="linear", align_corners=False)
    return x.transpose(1, 2).contiguous()

class VisionLLM(nn.Module):
    """
    Wraps a frozen vision encoder and frozen LLM.
    Trains a small adapter MLP and a cross-attention fusion in fp32 for stability.
    """
    def __init__(self, vis_model, vis_proc, llm, tok, llm_hidden: int, cfg: Dict[str, Any]):
        super().__init__()
        self.vision, self.proc, self.llm, self.tok = vis_model, vis_proc, llm, tok
        self.device = next(llm.parameters()).device

        # Infer vision feature dim
        dummy = Image.new("RGB", (384, 384), color=(120, 150, 180))
        v_dim = int(vision_tokens_from_pils(self.vision, self.proc, [dummy], self.device).shape[-1])

        self.cfg = FusionConfig(
            llm_hidden=llm_hidden,
            proj_hidden=cfg.get("proj_mlp_hidden", 4096),
            heads=cfg.get("fusion_heads", 8),
            max_vision_tokens=cfg.get("max_vision_tokens", 256),
            init_gate=cfg.get("init_gate", -2.0),
        )

        # Trainables are kept in fp32
        self.adapter = ProjectionAdapter(v_dim, llm_hidden, self.cfg.proj_hidden).to(self.device, dtype=torch.float32)
        self.fusion  = CrossAttentionFusion(llm_hidden, self.cfg.heads, self.cfg.init_gate).to(self.device, dtype=torch.float32)

        # Freeze backbone models
        for p in self.llm.parameters(): p.requires_grad = False
        for p in self.vision.parameters(): p.requires_grad = False
        for p in self.adapter.parameters(): p.requires_grad = True
        for p in self.fusion.parameters():  p.requires_grad = True

        # Token embedding to bypass tokenizer ids during fusion
        self.embed = self.llm.get_input_embeddings()

    def fuse(self, text_fp32: torch.Tensor, vis_fp32: torch.Tensor):
        v_proj = self.adapter(vis_fp32)
        v_proj = resample_seq(v_proj, self.cfg.max_vision_tokens)
        return self.fusion(text_fp32, v_proj)

    def forward_train(self, pils: list, prompts: list, targets: list):
        # Build inputs and label mask
        inputs, src_lens = [], []
        for p, y in zip(prompts, targets):
            s = p.strip(); t = y.strip()
            inputs.append(s + "\n" + t)
            src_lens.append(len(self.tok(s, add_special_tokens=False)["input_ids"]) + 1)

        enc = self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids  = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        labels = ids.clone()
        for i, L in enumerate(src_lens):
            labels[i, :L] = -100

        # Embeddings for fusion in fp32
        text_fp32 = self.embed(ids).to(torch.float32)
        vis_fp32  = vision_tokens_from_pils(self.vision, self.proc, pils, self.device).to(torch.float32)

        fused_fp32 = self.fuse(text_fp32, vis_fp32)
        fused = fused_fp32.to(next(self.llm.parameters()).dtype)  # back to LLM dtype

        out = self.llm(inputs_embeds=fused, attention_mask=attn, labels=labels)
        return out.loss, text_fp32, vis_fp32

    @torch.no_grad()
    def generate(self, pils: list, prompt: str, max_new_tokens: int = 48, generation_kwargs: Dict[str, Any] = None):
        """Run inference with optional sampling controls.

        Args:
            pils: List of PIL images to describe.
            prompt: Prompt string used for conditioning the LLM.
            max_new_tokens: Maximum number of tokens to decode if not overridden in
                ``generation_kwargs``.
            generation_kwargs: Optional dictionary passed directly to
                ``self.llm.generate`` allowing control over sampling behaviour.
        """

        self.eval()
        self.llm.eval()
        self.vision.eval()

        enc = self.tok(prompt, return_tensors="pt", padding=False)
        ids  = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        text_fp32 = self.embed(ids).to(torch.float32)
        vis_fp32  = vision_tokens_from_pils(self.vision, self.proc, pils, self.device).to(torch.float32)
        fused_fp32= self.fuse(text_fp32, vis_fp32)
        fused     = fused_fp32.to(next(self.llm.parameters()).dtype)

        gen_kwargs = dict(generation_kwargs or {})
        gen_kwargs.setdefault("max_new_tokens", max_new_tokens)
        gen_kwargs.setdefault("do_sample", False)
        if "pad_token_id" not in gen_kwargs and self.tok.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tok.pad_token_id
        if "eos_token_id" not in gen_kwargs and self.tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tok.eos_token_id

        gen_ids = self.llm.generate(inputs_embeds=fused, attention_mask=attn, **gen_kwargs)

        prompt_len = ids.shape[-1]
        output_ids = gen_ids[0]
        new_ids = output_ids[prompt_len:]
        if new_ids.numel() == 0:
            new_ids = output_ids
        return self.tok.decode(new_ids, skip_special_tokens=True).strip()

def load_models(vision_name: str, vision_fallback: str, llm_name: str, load_4bit: bool = True, cache_dir: str = None):
    # LLM in 4-bit
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(load_4bit),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    llm_conf = AutoConfig.from_pretrained(llm_name, trust_remote_code=True, cache_dir=cache_dir)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True, cache_dir=cache_dir)
    tok = AutoTokenizer.from_pretrained(llm_name, use_fast=True, trust_remote_code=True, cache_dir=cache_dir)
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    if getattr(tok, "bos_token_id", None) is None and tok.eos_token_id is not None:
        tok.bos_token_id = tok.eos_token_id

    llm_hidden = getattr(llm_conf, "hidden_size", None)
    for k in ["n_embd", "d_model", "model_dim"]:
        if llm_hidden is None and hasattr(llm_conf, k):
            llm_hidden = getattr(llm_conf, k)
    if llm_hidden is None:
        llm_hidden = 2048

    # Vision
    def _load_vis(name):
        proc = AutoProcessor.from_pretrained(name, trust_remote_code=True, cache_dir=cache_dir)
        mdl  = AutoModel.from_pretrained(name, trust_remote_code=True, cache_dir=cache_dir)
        return proc, mdl

    try:
        vis_proc, vis_model = _load_vis(vision_name)
    except Exception:
        vis_proc, vis_model = _load_vis(vision_fallback)

    for p in llm.parameters(): p.requires_grad = False
    llm.eval()
    for p in vis_model.parameters(): p.requires_grad = False
    vis_model.eval()

    return vis_proc, vis_model, llm, tok, llm_hidden
