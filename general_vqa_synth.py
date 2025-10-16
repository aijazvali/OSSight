# general_vqa_synth.py — robust general VQA synth with Caltech101 -> CIFAR100 fallback
import random
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

SYS = "You are a helpful vision assistant."
Q_SIMPLE = "What is in this image?"

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: [3,H,W] in [0,1]
    arr = (x.mul(255).clamp(0,255).permute(1,2,0).numpy()).astype(np.uint8)
    return Image.fromarray(arr)

def make_target(name: str) -> str:
    name = name.replace("_"," ").replace("-"," ")
    return f"It shows {name}."

class _BaseSynth(Dataset):
    def __init__(self):
        self.classes: List[str] = []

    def __len__(self): ...
    def __getitem__(self, i: int): ...

    @staticmethod
    def _prompt_and_target(label_name: str):
        prompt = f"{SYS}\nUser: {Q_SIMPLE}\nAssistant:"
        target = make_target(label_name)
        return prompt, target

class _Caltech101Synth(_BaseSynth):
    def __init__(self, root="/content/caltech101", resize=384, split="train",
                 seed=42, train_ratio=0.9):
        super().__init__()
        random.seed(seed)
        tfm = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        base = torchvision.datasets.Caltech101(root=root, download=True, transform=tfm)
        N = len(base)
        n_train = int(train_ratio * N)
        indices = list(range(N))
        random.Random(seed).shuffle(indices)
        self.indices = indices[:n_train] if split == "train" else indices[n_train:]
        self.base = base
        self.classes = base.categories

    def __len__(self): return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, label = self.base[idx]
        label_name = self.base.categories[label]
        prompt, target = self._prompt_and_target(label_name)
        return img, prompt, target, label_name

class _CIFAR100Synth(_BaseSynth):
    def __init__(self, root="/content/cifar100", resize=384, split="train",
                 seed=42, train_ratio=0.9):
        super().__init__()
        random.seed(seed)
        tfm = transforms.Compose([
            transforms.Resize((resize, resize), antialias=True),
            transforms.ToTensor(),
        ])
        base = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
        # We'll ignore the small "test" split of CIFAR100 and carve our own train/val
        N = len(base)
        n_train = int(train_ratio * N)
        indices = list(range(N))
        random.Random(seed).shuffle(indices)
        self.indices = indices[:n_train] if split == "train" else indices[n_train:]
        self.base = base
        self.classes = base.classes  # list of 100 fine-grained classes

    def __len__(self): return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, label = self.base[idx]
        label_name = self.classes[label]
        prompt, target = self._prompt_and_target(label_name)
        return img, prompt, target, label_name

def _collate_common(batch):
    imgs, prompts, targets, labels = [], [], [], []
    for img_t, p, t, ln in batch:
        imgs.append(tensor_to_pil(img_t))
        prompts.append(p); targets.append(t); labels.append(ln)
    return imgs, prompts, targets, labels

# Public API (same names used by train_multi.py)
class Caltech101VQASynth(Dataset):
    """
    Wrapper that prefers Caltech101 and falls back to CIFAR100 automatically if the
    Caltech101 download fails (e.g., HTTP 404).
    """
    def __init__(self, root="/content/caltech101", resize=384, split="train",
                 seed=42, train_ratio=0.9):
        try:
            self._impl = _Caltech101Synth(root, resize, split, seed, train_ratio)
            self.is_fallback = False
        except Exception as e:
            print("[general_vqa_synth] Caltech101 unavailable, falling back to CIFAR100. Reason:", repr(e))
            self._impl = _CIFAR100Synth("/content/cifar100", resize, split, seed, train_ratio)
            self.is_fallback = True
        self.classes = getattr(self._impl, "classes", [])

    def __len__(self): return len(self._impl)
    def __getitem__(self, i: int): return self._impl[i]

def collate(batch):
    return _collate_common(batch)
