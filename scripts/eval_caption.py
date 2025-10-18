import argparse, random, sys
from pathlib import Path
from typing import Optional
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ossight.data import COCODataset, default_prompt
from ossight.model import load_models, VisionLLM


def resolve_path(path_str: Optional[str]) -> Path:
    if path_str is None:
        raise ValueError("Expected a path string, got None")
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_path_exists(path: Path, *, description: str, expect_dir: bool) -> None:
    if expect_dir:
        if not path.is_dir():
            raise FileNotFoundError(f"{description} directory not found: {path}")
    else:
        if not path.is_file():
            raise FileNotFoundError(f"{description} file not found: {path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-images", type=str, required=True)
    ap.add_argument("--coco-captions", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--prompt", type=str, default=None,
                    help="Override the default instruction prompt used during captioning.")
    ap.add_argument("--query", type=str, default=None,
                    help="Optional user query appended to the instruction prompt before decoding.")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    ap.add_argument("--hf-cache", type=str, default=None)
    ap.add_argument("--vision", type=str, default="google/siglip-so400m-patch14-384")
    ap.add_argument("--vision-fallback", type=str, default="google/siglip-base-patch16-384")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.samples <= 0:
        raise ValueError(f"--samples must be positive, got {args.samples}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    coco_images = resolve_path(args.coco_images)
    coco_captions = resolve_path(args.coco_captions)
    ckpt_path = resolve_path(args.ckpt)
    ensure_path_exists(coco_images, description="COCO image root", expect_dir=True)
    ensure_path_exists(coco_captions, description="COCO captions", expect_dir=False)
    ensure_path_exists(ckpt_path, description="Checkpoint", expect_dir=False)

    cache_dir = resolve_path(args.hf_cache) if args.hf_cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    vis_proc, vis_model, llm, tok, llm_hidden = load_models(
        args.vision, args.vision_fallback, args.llm, load_4bit=True, cache_dir=str(cache_dir) if cache_dir else None
    )
    cfg = {
        "proj_mlp_hidden": 4096,
        "fusion_heads": 8,
        "max_vision_tokens": 256,
        "init_gate": -2.0,
    }
    wrapper = VisionLLM(vis_model, vis_proc, llm, tok, llm_hidden, cfg)
    state = torch.load(str(ckpt_path), map_location="cpu")
    wrapper.adapter.load_state_dict(state["adapter"])
    wrapper.fusion.load_state_dict(state["fusion"])
    wrapper = wrapper.to(next(llm.parameters()).device)
    wrapper.eval()

    ds = COCODataset(str(coco_images), str(coco_captions), limit=None)
    if len(ds) == 0:
        raise ValueError(
            "COCO dataset is empty. Ensure the captions file and image directory correspond to the same split."
        )
    idxs = random.sample(range(len(ds)), k=min(args.samples, len(ds)))
    prompt_text = args.prompt if args.prompt is not None else default_prompt()

    def compose_prompt(base: str) -> str:
        base_clean = base.strip()
        if args.query is None or not args.query.strip():
            return base_clean
        query_clean = args.query.strip()
        if base_clean:
            return f"{base_clean}\n{query_clean}"
        return query_clean
    gen_kwargs = {}
    if args.temperature is not None:
        if args.temperature <= 0:
            raise ValueError("--temperature must be positive when provided")
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        if not (0.0 < args.top_p <= 1.0):
            raise ValueError("--top-p must be within (0, 1]")
        gen_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        if args.top_k < 0:
            raise ValueError("--top-k must be non-negative")
        gen_kwargs["top_k"] = args.top_k
    if args.do_sample or any(k in gen_kwargs for k in ("temperature", "top_p", "top_k")):
        gen_kwargs["do_sample"] = True
    gen_kwargs["max_new_tokens"] = args.max_new_tokens

    for i, idx in enumerate(idxs, 1):
        ex = ds[idx]
        base_prompt = prompt_text if args.prompt is not None else ex["prompt"]
        prompt_to_use = compose_prompt(base_prompt)
        pred = wrapper.generate([ex["pil"]], prompt_to_use, generation_kwargs=gen_kwargs)
        print(f"[{i}] GT:   {ex['target']}")
        print(f"    PRED: {pred}\n")

if __name__ == "__main__":
    main()
