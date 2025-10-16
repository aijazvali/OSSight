# train_multi.py
import argparse, os, time, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm

from mm_modules import MMConfig, VisionMiniVLM
from eurosat_vqa import get_datasets as get_eo_datasets, collate_eo, make_labels, QUESTION as EO_Q
from general_vqa_synth import Caltech101VQASynth, collate as collate_gen

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate_fast(model, val_eo_loader, val_gen_loader, cap_batches=32):
    model.eval(); tot=0.0; n=0
    with torch.no_grad():
        # EO
        pbar = tqdm(val_eo_loader, total=min(len(val_eo_loader), cap_batches), desc="Validating EO", leave=False)
        for step, (imgs, prompts, targets, _) in enumerate(pbar):
            if step>=cap_batches: break
            input_ids, attn, labels = make_labels(model.tokenizer, prompts, targets, model.device)
            out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
            loss = float(out.loss.detach()); tot+=loss; n+=1; pbar.set_postfix(loss=loss)
        # General
        pbar = tqdm(val_gen_loader, total=min(len(val_gen_loader), cap_batches), desc="Validating GEN", leave=False)
        for step, (imgs, prompts, targets, _) in enumerate(pbar):
            if step>=cap_batches: break
            tok_full = model.tokenizer([s+" "+t for s,t in zip(prompts,targets)],
                                       padding=True, truncation=True, return_tensors="pt")
            tok_src  = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
            input_ids = tok_full["input_ids"].to(model.device)
            attn = tok_full["attention_mask"].to(model.device)
            labels = input_ids.clone()
            for i in range(input_ids.size(0)):
                src_len = int(tok_src["attention_mask"][i].sum().item())
                labels[i, :src_len] = -100
            out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
            loss = float(out.loss.detach()); tot+=loss; n+=1; pbar.set_postfix(loss=loss)
    model.train()
    return tot/max(n,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps_per_epoch", type=int, default=300)
    ap.add_argument("--max_vision_tokens", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_dir", default="./mini_vlm_ckpt_multi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix_ratio", type=float, default=0.5, help="fraction of EO batches per epoch")
    args = ap.parse_args()

    set_seed(args.seed)

    # EO data
    ds_all, train_eo, val_eo = get_eo_datasets()
    class_list = ds_all.class_names
    train_eo_loader = DataLoader(train_eo, batch_size=args.batch_size, shuffle=True, collate_fn=collate_eo)
    val_eo_loader   = DataLoader(val_eo,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_eo)

    # General data
    train_gen = Caltech101VQASynth(split="train")
    val_gen   = Caltech101VQASynth(split="val")
    train_gen_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True, collate_fn=collate_gen)
    val_gen_loader   = DataLoader(val_gen,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_gen)

    # Model
    cfg = MMConfig(vision_model_name="google/siglip-so400m-patch14-384",
                   llm_name_or_path=args.llm, dtype=args.dtype, use_4bit=True,
                   max_vision_tokens=args.max_vision_tokens, num_heads=-1)
    model = VisionMiniVLM(cfg)
    for p in model.adapter.parameters(): p.requires_grad = True
    for p in model.fusion.parameters():  p.requires_grad = True

    optimizer = optim.AdamW(list(model.adapter.parameters())+list(model.fusion.parameters()),
                            lr=args.lr, weight_decay=0.0)
    use_cuda = (model.device.type=='cuda')
    mp_dtype = torch.bfloat16 if args.dtype=="bfloat16" else torch.float16
    use_scaler = use_cuda and (mp_dtype==torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # Iterators
    it_eo = cycle(train_eo_loader)
    it_gen = cycle(train_gen_loader)
    n_eo = int(args.steps_per_epoch * args.mix_ratio)
    n_gen = args.steps_per_epoch - n_eo

    model.train()
    for ep in range(args.epochs):
        t0 = time.time()
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {ep+1}")
        for step in pbar:
            # alternate by ratio
            use_eo = (step < n_eo) or (step % 2 == 0)
            if use_eo:
                imgs, prompts, targets, _ = next(it_eo)
                input_ids, attn, labels = make_labels(model.tokenizer, prompts, targets, model.device)
            else:
                imgs, prompts, targets, _ = next(it_gen)
                src = prompts; tgt = [" "+t.strip() for t in targets]
                full = [s+t for s,t in zip(src,tgt)]
                tok_full = model.tokenizer(full, padding=True, truncation=True, return_tensors="pt")
                tok_src  = model.tokenizer(src,  padding=True, truncation=True, return_tensors="pt")
                input_ids = tok_full["input_ids"].to(model.device)
                attn = tok_full["attention_mask"].to(model.device)
                labels = input_ids.clone()
                for i in range(input_ids.size(0)):
                    src_len = int(tok_src["attention_mask"][i].sum().item())
                    labels[i, :src_len] = -100

            optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.autocast("cuda", dtype=mp_dtype):
                    out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
                    loss = out.loss
                if use_scaler:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            else:
                out = model(images=imgs, input_ids=input_ids, labels=labels, attention_mask=attn)
                loss = out.loss
                loss.backward(); optimizer.step()

            pbar.set_postfix(loss=float(loss.detach()))

        val_loss = evaluate_fast(model, val_eo_loader, val_gen_loader, cap_batches=32)
        print(f"Epoch {ep+1} | Val(mix) loss: {val_loss:.4f} | time {time.time()-t0:.1f}s")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.adapter.state_dict(), f"{args.save_dir}/adapter.pt")
    torch.save(model.fusion.state_dict(),  f"{args.save_dir}/fusion.pt")
    print("Saved:", args.save_dir)

if __name__ == "__main__":
    main()
