# OSSight: Minimal Multimodal Vision–Language Prototype  
**SigLIP + Qwen2.5-1.5B | Lightweight Adapter + Fusion**

---

### Overview
OSSight is a minimal, end-to-end multimodal research prototype that connects a frozen **SigLIP vision encoder** with a frozen **Qwen2.5-1.5B-Instruct** large language model through a lightweight **adapter-fusion module** trained in fp32 precision.  

The project demonstrates a practical pipeline for aligning image and text embeddings using simple losses and limited hardware — it runs comfortably on a single **NVIDIA T4** (e.g., Colab or Kaggle).

---

### Key Features
- **Architecture:** SigLIP → Adapter MLP → Cross-Attention Fusion → Qwen2.5-1.5B (4-bit)  
- **Training objective:** Cross-entropy caption loss + cosine alignment loss  
- **Stable precision mix:** Vision & LLM in 4-bit/fp16, fusion in fp32  
- **Dataset:** COCO captions (val2017 or train2017)  
- **Frameworks:** PyTorch + Hugging Face Transformers + BitsAndBytes  
- **Output:** Caption generation and vision-language grounding prototype

---

## Installation

```bash
git clone <your-repo-url> ossight
cd ossight
python -m venv .venv && source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Setup

To train or evaluate, download COCO images and captions:

```bash
python scripts/download_coco.py --root ./data --split val2017
```

This creates:
```
data/
 └─ coco/
     ├─ images/val2017/
     └─ annotations/captions_val2017.json
```

For larger training runs:
```bash
python scripts/download_coco.py --root ./data --split train2017
```
*(train2017 ≈ 18 GB)*

---

## Training

Run the base alignment stage (**S0**):

```bash
python scripts/train_s0.py   --coco-images ./data/coco/images/val2017   --coco-captions ./data/coco/annotations/captions_val2017.json   --out ./checkpoints   --steps 1200   --batch-size 4   --grad-accum 4   --lr 3e-5   --alpha-align 0.1
```

This trains only the **adapter + fusion** modules while keeping both SigLIP and Qwen frozen.

---

## Evaluation

Run a quick greedy caption generation test:

```bash
python scripts/eval_caption.py   --coco-images ./data/coco/images/val2017   --coco-captions ./data/coco/annotations/captions_val2017.json   --ckpt ./checkpoints/adapter_fusion_step1200.pt   --samples 8
```

---

## How to Run on Kaggle or Colab

1. Enable **Internet access** in the environment.  
2. Execute the notebook cells from the repository’s `notebooks/` folder or directly import the training scripts.  
3. Checkpoints and cache will automatically save under `/kaggle/working` or `/content/drive`.

---

## Roadmap and Future Work

### **Stage S1 — Extended Caption Grounding**
**Goal:** Strengthen visual grounding and reduce language bias.  
**Planned Steps:**
- Expand to full **COCO train2017** and include **Conceptual Captions 3M**.  
- Add **region-level supervision** using COCO bounding boxes.  
- Introduce **LoRA fine-tuning** on selected Qwen attention layers for better cross-modal coupling.

---

### **Stage S2 — Cross-Domain Generalization**
**Goal:** Adapt the model for geospatial and scientific imagery.  
**Planned Steps:**
- Train on **BigEarthNet** or **Sentinel-2** image–text pairs.  
- Replace or supplement SigLIP with **CLIP-ViT-L** or **Prithvi EO encoder**.  
- Develop a **dual-head architecture** for environmental question answering and domain-specific reasoning.

---

### **Planned Enhancements**
- Add **VQA-style fine-tuning** (question–answer pairs).  
- Include **evaluation scripts** for BLEU, CIDEr, and CLIPScore.  
- Integrate **gradient checkpointing** for larger batches on limited GPUs.  
- Experiment with **temporal fusion** for short video captioning.  
- Add **multi-modal instruction tuning** for broader generalization.

---

## Typical Hyperparameters

| Parameter | Recommended |
|------------|--------------|
| Base learning rate | 3e-5 |
| Gradient accumulation | 4 |
| Batch size | 4 |
| Alpha (cosine weight) | 0.1 |
| Max steps | 1200–4800 |
| Precision | fp32 (fusion) + 4-bit (LLM) |

---

## Example Outputs

| Image | Ground Truth Caption | Model Output |
|--------|----------------------|---------------|
| Kitchen scene | "A kitchen with a stove top oven next to a white fridge." | "A modern kitchen with a stove and refrigerator." |
| Plane | "A plane flying in the sky with a trail of white smoke." | "An airplane leaving a white trail in the blue sky." |

---
