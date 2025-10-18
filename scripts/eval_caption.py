import os, argparse, random, json, sys
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm.auto import tqdm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ossight.data import COCODataset
from ossight.model import load_models, VisionLLM


def resolve_path(path_str: Optional[str]) -> Path:
    if path_str is None:
        raise ValueError("Expected a path string, got None")
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-images", type=str, required=True)
    ap.add_argument("--coco-captions", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--hf-cache", type=str, default=None)
    ap.add_argument("--vision", type=str, default="google/siglip-so400m-patch14-384")
    ap.add_argument("--vision-fallback", type=str, default="google/siglip-base-patch16-384")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    return ap.parse_args()

def main():
    args = parse_args()
    coco_images = resolve_path(args.coco_images)
    coco_captions = resolve_path(args.coco_captions)
    ckpt_path = resolve_path(args.ckpt)
    cache_dir = resolve_path(args.hf_cache) if args.hf_cache else None
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

    ds = COCODataset(str(coco_images), str(coco_captions), limit=None)
    idxs = random.sample(range(len(ds)), k=min(args.samples, len(ds)))
    for i, idx in enumerate(idxs, 1):
        ex = ds[idx]
        pred = wrapper.generate([ex["pil"]], "Describe the image briefly.", max_new_tokens=48)
        print(f"[{i}] GT:   {ex['target']}")
        print(f"    PRED: {pred}\n")

if __name__ == "__main__":
    main()
