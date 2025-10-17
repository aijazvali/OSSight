# OSSight: Minimal Multimodal Prototype (SigLIP + Qwen2.5-1.5B)

A small, practical vision-language prototype:
- Vision: SigLIP (frozen)
- LLM: Qwen2.5-1.5B-Instruct (4-bit, frozen)
- Trainable: Adapter MLP + Cross-Attention Fusion (fp32)
- Loss: Captioning cross-entropy + cosine alignment (for grounding)
- Data: COCO captions

This repo is designed to run on a single T4-class GPU (Colab, Kaggle, or local).

---

## Quickstart

### 1) Environment
```bash
git clone <your-repo-url> ossight
cd ossight
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
