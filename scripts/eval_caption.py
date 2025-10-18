import os, argparse, random, json, sys
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ossight.data import COCODataset
from ossight.model import load_models, VisionLLM

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
    vis_proc, vis_model, llm, tok, llm_hidden = load_models(
        args.vision, args.vision_fallback, args.llm, load_4bit=True, cache_dir=args.hf_cache
    )
    cfg = {
        "proj_mlp_hidden": 4096,
        "fusion_heads": 8,
        "max_vision_tokens": 256,
        "init_gate": -2.0,
    }
    wrapper = VisionLLM(vis_model, vis_proc, llm, tok, llm_hidden, cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    wrapper.adapter.load_state_dict(state["adapter"])
    wrapper.fusion.load_state_dict(state["fusion"])
    wrapper = wrapper.to(next(llm.parameters()).device)

    ds = COCODataset(args.coco_images, args.coco_captions, limit=None)
    idxs = random.sample(range(len(ds)), k=min(args.samples, len(ds)))
    for i, idx in enumerate(idxs, 1):
        ex = ds[idx]
        pred = wrapper.generate([ex["pil"]], "Describe the image briefly.", max_new_tokens=48)
        print(f"[{i}] GT:   {ex['target']}")
        print(f"    PRED: {pred}\n")

if __name__ == "__main__":
    main()
