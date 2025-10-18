"""Launch a Gradio demo for the OSSight captioning prototype."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import gradio as gr
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from ossight.data import default_prompt
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the OSSight Gradio captioning demo.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to adapter/fusion checkpoint.")
    ap.add_argument("--vision", type=str, default="google/siglip-so400m-patch14-384")
    ap.add_argument("--vision-fallback", type=str, default="google/siglip-base-patch16-384")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--hf-cache", type=str, default=None, help="Optional Hugging Face cache directory.")
    ap.add_argument("--prompt", type=str, default=None, help="Base instruction prompt shown in the UI.")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Host address for Gradio server.")
    ap.add_argument("--port", type=int, default=7860, help="Port for Gradio server.")
    ap.add_argument("--share", action="store_true", help="Enable Gradio public sharing tunnel.")
    ap.add_argument("--no-4bit", action="store_true", help="Load the language model in full precision.")
    return ap.parse_args()


def load_wrapper(args: argparse.Namespace) -> VisionLLM:
    ckpt_path = resolve_path(args.ckpt)
    ensure_path_exists(ckpt_path, description="Checkpoint", expect_dir=False)

    cache_dir = resolve_path(args.hf_cache) if args.hf_cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    vis_proc, vis_model, llm, tok, llm_hidden = load_models(
        args.vision,
        args.vision_fallback,
        args.llm,
        load_4bit=not args.no_4bit,
        cache_dir=str(cache_dir) if cache_dir else None,
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
    wrapper.eval()
    return wrapper


def compose_prompt(base: str, query: Optional[str]) -> str:
    base_clean = (base or "").strip()
    if query is None or not query.strip():
        return base_clean
    query_clean = query.strip()
    if base_clean:
        return f"{base_clean}\n{query_clean}"
    return query_clean


def create_interface(wrapper: VisionLLM, base_prompt: str) -> gr.Blocks:
    description = (
        "Upload an image, optionally provide a query, and the OSSight adapter will "
        "generate a grounded response using the frozen SigLIP vision encoder and Qwen LLM."
    )

    def infer(image: np.ndarray, query: str, max_new_tokens: int, temperature: float,
              top_p: float, top_k: int, do_sample: bool) -> str:
        if image is None:
            return "Please upload an image before running inference."
        pil = Image.fromarray(image.astype("uint8"))
        prompt = compose_prompt(base_prompt, query)
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
        if do_sample:
            gen_kwargs["do_sample"] = True
        if temperature is not None:
            if temperature <= 0:
                return "Temperature must be positive."
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            if not (0 < top_p <= 1):
                return "Top-p must be within (0, 1]."
            gen_kwargs["top_p"] = float(top_p)
            gen_kwargs.setdefault("do_sample", True)
        if top_k is not None:
            if top_k < 0:
                return "Top-k must be non-negative."
            gen_kwargs["top_k"] = int(top_k)
            gen_kwargs.setdefault("do_sample", True)

        try:
            output = wrapper.generate([pil], prompt, generation_kwargs=gen_kwargs)
        except Exception as exc:  # pragma: no cover - defensive against runtime GPU/IO issues
            return f"Generation failed: {exc}"
        return output

    with gr.Blocks(title="OSSight Captioning Demo") as demo:
        gr.Markdown("# OSSight Captioning Demo")
        gr.Markdown(description)
        if base_prompt:
            gr.Markdown(f"**Base instruction:** {base_prompt}")

        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Input image")
                query = gr.Textbox(label="User query", placeholder="Describe what you'd like to know about the image.")
                run_button = gr.Button("Generate caption", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Model output", lines=8)

        with gr.Accordion("Generation settings", open=False):
            max_new_tokens = gr.Slider(label="Max new tokens", minimum=16, maximum=256, value=96, step=8)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.7, step=0.05, interactive=True)
            top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=0.9, step=0.05, interactive=True)
            top_k = gr.Slider(label="Top-k", minimum=0, maximum=200, value=40, step=5, interactive=True)
            do_sample = gr.Checkbox(label="Enable sampling", value=True)

        def _wrap_infer(image, query, max_new_tokens, temperature, top_p, top_k, do_sample):
            temp = float(temperature) if temperature is not None else None
            tp = float(top_p) if top_p is not None and top_p > 0 else None
            tk = int(top_k) if top_k is not None else None
            return infer(image, query, max_new_tokens, temp, tp, tk, do_sample)

        inputs = [image, query, max_new_tokens, temperature, top_p, top_k, do_sample]
        run_button.click(fn=_wrap_infer, inputs=inputs, outputs=[output])

        gr.Examples(
            examples=[
                ["https://huggingface.co/datasets/hf-internal-testing/example-images/resolve/main/dog.png", "Describe the scene"],
                ["https://huggingface.co/datasets/hf-internal-testing/example-images/resolve/main/beach.png", "What is the weather like?"],
            ],
            inputs=[image, query],
            cache_examples=False,
        )

    return demo


def main():
    args = parse_args()
    base_prompt = args.prompt if args.prompt is not None else default_prompt()
    wrapper = load_wrapper(args)
    demo = create_interface(wrapper, base_prompt)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
