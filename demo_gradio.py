# demo_gradio.py — Demo UI with mode switch (General VQA vs EO concise)
import argparse
import torch
import gradio as gr

from mm_modules import MMConfig, VisionMiniVLM
from eurosat_vqa import get_datasets, QUESTION as EO_QUESTION

def build_model(llm="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16",
                max_vision_tokens=64, ckpt_dir="./mini_vlm_ckpt_multi"):
    cfg = MMConfig(
        vision_model_name="google/siglip-so400m-patch14-384",
        llm_name_or_path=llm,
        dtype=dtype,
        use_4bit=True,
        max_vision_tokens=max_vision_tokens,
        num_heads=-1,
    )
    model = VisionMiniVLM(cfg)
    # Load tuned adapter/fusion if available
    try:
        adapter_path = f"{ckpt_dir}/adapter.pt"
        fusion_path  = f"{ckpt_dir}/fusion.pt"
        model.adapter.load_state_dict(torch.load(adapter_path, map_location=model.device))
        model.fusion.load_state_dict(torch.load(fusion_path, map_location=model.device))
        print(f"Loaded fine-tuned weights from {ckpt_dir}")
    except Exception as e:
        print("Warning: could not load fine-tuned weights:", e)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"])
    ap.add_argument("--max_vision_tokens", type=int, default=64)
    ap.add_argument("--ckpt_dir", default="./mini_vlm_ckpt_multi")
    args = ap.parse_args()

    model = build_model(args.llm, args.dtype, args.max_vision_tokens, args.ckpt_dir)
    ds_all, _, _ = get_datasets()
    class_list = ds_all.class_names

    def infer(image, question, mode):
        if image is None or not str(question).strip():
            return "Please provide an image and a question."
        if mode == "EO (concise)":
            # EO short answer and label mapping
            q = question or EO_QUESTION
            text = model.generate_landcover([image], q, max_new_tokens=24, min_new_tokens=6)
            label = model.predict_label([image], q, class_list)
            return f"{text}\n\n[Predicted label: {label}]"
        else:
            # General VQA free-form
            return model.generate_freeform([image], question, max_new_tokens=96, min_new_tokens=8, temperature=0.5, top_p=0.9)

    demo = gr.Interface(
        fn=infer,
        inputs=[
            gr.Image(type="pil"),
            gr.Textbox(label="Question", value="What is happening in this image?"),
            gr.Dropdown(choices=["General VQA", "EO (concise)"], value="General VQA", label="Mode")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="SigLIP + CrossAttention + Qwen-1.5B (4-bit)",
        description="Switch between General VQA (free-form) and EO land-cover mode.",
    )
    demo.launch(debug=False, share=True)

if __name__ == "__main__":
    main()
