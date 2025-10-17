from typing import Dict, List, Tuple
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader
import json, os, random

def _resize_flip_blur(pil: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = pil.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    if random.random() < 0.5:
        pil = ImageOps.mirror(pil)
    if random.random() < 0.1:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    return pil

def default_prompt() -> str:
    return (
        "You are a precise vision assistant.\n"
        "Task: Describe the image in one short sentence mentioning the main objects and scene.\n"
        "Answer:"
    )

class COCODataset(Dataset):
    """
    Minimal COCO captions loader. Expects a folder of images and a captions JSON.
    """
    def __init__(self, img_dir: str, captions_json: str, max_side: int = 512, limit: int = None):
        with open(captions_json, "r") as f:
            coco = json.load(f)
        id2file = {img["id"]: img["file_name"] for img in coco["images"]}
        samples = []
        for ann in coco["annotations"]:
            p = os.path.join(img_dir, id2file.get(ann["image_id"], ""))
            if os.path.exists(p):
                samples.append((p, ann["caption"]))
        if limit:
            samples = samples[:limit]
        self.samples = samples
        self.max_side = max_side

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        p, cap = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = _resize_flip_blur(img, self.max_side)
        return {"pil": img, "prompt": default_prompt(), "target": cap}

def coco_collate(batch: List[Dict]) -> Dict[str, List]:
    return {
        "pils":    [x["pil"] for x in batch],
        "prompts": [x["prompt"] for x in batch],
        "targets": [x["target"] for x in batch],
    }

def build_loader(img_dir: str, captions_json: str, batch_size: int, max_side: int = 512, limit: int = None, workers: int = 2):
    ds = COCODataset(img_dir, captions_json, max_side=max_side, limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=coco_collate, drop_last=True, num_workers=workers), ds
