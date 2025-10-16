# train_eurosat.py
import argparse, os, time, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mm_modules import MMConfig, VisionMiniVLM
from eurosat_vqa import get_datasets, collate_eo, make_labels, QUESTION

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_fast(model: VisionMiniVLM, val_loader, cap_batches=64):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        pbar = tqdm(val_loader, total=min(len(val_loader), cap_batches), desc="Validating", leave=False)
        for step, (imgs, prompts, targets, _) in enumerate(pbar):
            if step >= cap_batches: break
            input_ids, attn, labels = make_labels(model.tokenizer, prompts, targets, model.device)
            out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
            loss = float(out.loss.detach())
            tot += loss; n += 1
            pbar.set_postfix(loss=loss)
    model.train()
    return tot / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps_per_epoch", type=int, default=200)     # cap for free tier
    ap.add_argument("--max_vision_tokens", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_dir", default="./mini_vlm_ckpt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # Datasets / loaders
    ds_all, train_set, val_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_eo)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_eo)

    # Model
    cfg = MMConfig(
        vision_model_name="google/siglip-so400m-patch14-384",
        llm_name_or_path=args.llm,
        dtype=args.dtype,
        use_4bit=True,
        max_vision_tokens=args.max_vision_tokens,
        num_heads=-1,
    )
    model = VisionMiniVLM(cfg)
    print("Device:", model.device, "| H_dim:", model.h_dim)

    # Trainable: adapter + fusion only
    for p in model.adapter.parameters(): p.requires_grad = True
    for p in model.fusion.parameters():  p.requires_grad = True

    optim_groups = list(model.adapter.parameters()) + list(model.fusion.parameters())
    optimizer = optim.AdamW(optim_groups, lr=args.lr, weight_decay=0.0)

    use_cuda = (model.device.type == 'cuda')
    mp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    use_scaler = use_cuda and (mp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    model.train()
    for ep in range(args.epochs):
        t0 = time.time()
        pbar = tqdm(train_loader, total=min(len(train_loader), args.steps_per_epoch), desc=f"Epoch {ep+1}")
        for step, (imgs, prompts, targets, _) in enumerate(pbar):
            if step >= args.steps_per_epoch: break
            input_ids, attn, labels = make_labels(model.tokenizer, prompts, targets, model.device)
            optimizer.zero_grad(set_to_none=True)

            if use_cuda:
                with torch.autocast("cuda", dtype=mp_dtype):
                    out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
                    loss = out.loss
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
                loss = out.loss
                loss.backward()
                optimizer.step()

            pbar.set_postfix(loss=float(loss.detach()))
        val_loss = evaluate_fast(model, val_loader, cap_batches=64)
        print(f"Epoch {ep+1} done in {time.time()-t0:.1f}s | Val loss: {val_loss:.4f}")

    # Save trainables
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.adapter.state_dict(), os.path.join(args.save_dir, "adapter.pt"))
    torch.save(model.fusion.state_dict(),  os.path.join(args.save_dir, "fusion.pt"))
    print("Saved:", args.save_dir)

if __name__ == "__main__":
    main()
