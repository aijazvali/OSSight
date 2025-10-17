import os, argparse, yaml, math
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from ossight.utils import set_seed, ensure_dir, device
from ossight.data import build_loader
from ossight.model import load_models, VisionLLM

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-images", type=str, required=True)
    ap.add_argument("--coco-captions", type=str, required=True)
    ap.add_argument("--out", type=str, default="./checkpoints")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--alpha-align", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-cache", type=str, default=None)

    # Models
    ap.add_argument("--vision", type=str, default="google/siglip-so400m-patch14-384")
    ap.add_argument("--vision-fallback", type=str, default="google/siglip-base-patch16-384")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--no-4bit", action="store_true", help="Load LLM in full precision (not recommended on small GPUs).")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out)
    dev = device()
    load_4bit = not args.no_4bit

    # Data
    loader, _ = build_loader(args.coco_images, args.coco_captions, batch_size=args.batch_size)

    # Models
    vis_proc, vis_model, llm, tok, llm_hidden = load_models(
        vision_name=args.vision,
        vision_fallback=args.vision_fallback,
        llm_name=args.llm,
        load_4bit=load_4bit,
        cache_dir=args.hf_cache
    )
    cfg = {
        "proj_mlp_hidden": 4096,
        "fusion_heads": 8,
        "max_vision_tokens": 256,
        "init_gate": -2.0,
    }
    wrapper = VisionLLM(vis_model, vis_proc, llm, tok, llm_hidden, cfg).to(dev)

    # Trainables
    params = list(wrapper.adapter.parameters()) + list(wrapper.fusion.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_schedule(step):
        return args.lr * (step + 1) / max(1, args.warmup) if step < args.warmup else args.lr

    steps = args.steps
    grad_acc = args.grad_acc
    log_every = 20
    ckpt_every = 300
    alpha = args.alpha_align

    wrapper.train()
    global_step = 0
    run_ce = run_al = 0.0
    pbar = tqdm(total=steps, desc="Training S0 (COCO captions)")

    data_iter = iter(loader)
    while global_step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)

        loss_ce, text_fp32, vis_fp32 = wrapper.forward_train(batch["pils"], batch["prompts"], batch["targets"])
        # Cosine alignment term
        v_proj = wrapper.adapter(vis_fp32)
        t_mean = text_fp32.mean(dim=1)
        v_mean = v_proj.mean(dim=1)
        sim = torch.cosine_similarity(t_mean, v_mean, dim=-1)
        align = (1 - sim).mean()
        loss = loss_ce + alpha * align

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            print(f"non-finite loss at step {global_step}; skipping")
            pbar.update(1); global_step += 1; continue

        (loss / grad_acc).backward()
        if (global_step + 1) % grad_acc == 0:
            for g in optimizer.param_groups:
                g["lr"] = lr_schedule(global_step)
            clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        run_ce += float(loss_ce.detach())
        run_al += float(align.detach())
        if (global_step + 1) % log_every == 0:
            pbar.set_postfix({"CE": f"{run_ce/log_every:.3f}", "Align": f"{run_al/log_every:.3f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
            run_ce = run_al = 0.0

        if (global_step + 1) % ckpt_every == 0:
            ckpt = os.path.join(args.out, f"adapter_fusion_step{global_step+1}.pt")
            torch.save({
                "adapter": wrapper.adapter.state_dict(),
                "fusion":  wrapper.fusion.state_dict(),
                "step": global_step+1,
                "cfg": cfg
            }, ckpt)
            print(f"Saved checkpoint: {ckpt}")

        pbar.update(1); global_step += 1

    pbar.close()
    print("Done.")

if __name__ == "__main__":
    main()
