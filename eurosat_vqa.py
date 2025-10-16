# eurosat_vqa.py
import random
import numpy as np
from typing import Tuple, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

SYSTEM_INST = "You are a helpful EO (Earth-Observation) assistant."
QUESTION = "What is the main land cover in this satellite tile?"

def make_target(label_name: str) -> str:
    return f"The main land cover is {label_name.replace('_', ' ')}."

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: [3,H,W] in [0,1]
    arr = (x.mul(255).clamp(0,255).permute(1,2,0).numpy()).astype(np.uint8)
    return Image.fromarray(arr)

class EOQADataset(Dataset):
    def __init__(self, base: torchvision.datasets.VisionDataset):
        self.base = base
        self.class_names = base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        label_name = self.class_names[label]
        prompt = f"{SYSTEM_INST}\nUser: {QUESTION}\nAssistant:"
        target = make_target(label_name)
        return img, prompt, target, label_name

def collate_eo(batch) -> Tuple[List[Image.Image], List[str], List[str], List[str]]:
    imgs, prompts, targets, labels = [], [], [], []
    for img_t, p, t, ln in batch:
        imgs.append(tensor_to_pil(img_t))
        prompts.append(p)
        targets.append(t)
        labels.append(ln)
    return imgs, prompts, targets, labels

def make_labels(tokenizer, prompts: List[str], targets: List[str], device: torch.device):
    src = [s.rstrip() for s in prompts]
    tgt = [(" " + t.strip()) for t in targets]
    full = [s + " " + t for s,t in zip(src, tgt)]
    tok_full = tokenizer(full, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
    tok_src  = tokenizer(src,  padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)
    input_ids = tok_full["input_ids"].to(device)
    attention_mask = tok_full["attention_mask"].to(device)
    labels = input_ids.clone()
    # mask prompt portion (loss only on target)
    for i in range(len(src)):
        src_len = int(tok_src["attention_mask"][i].sum().item())
        labels[i, :src_len] = -100
    return input_ids, attention_mask, labels

def get_datasets(root="/content/eurosat", resize=384, train_ratio=0.9, seed=42):
    random.seed(seed)
    transform_384 = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ])
    base = torchvision.datasets.EuroSAT(root=root, download=True, transform=transform_384)
    ds = EOQADataset(base)
    N = len(ds)
    train_sz = int(train_ratio * N)
    val_sz = N - train_sz
    train_set, val_set = random_split(ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(seed))
    return ds, train_set, val_set
