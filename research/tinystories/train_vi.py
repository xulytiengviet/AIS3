"""Train or fine-tune a Vietnamese PLE TinyLM compatible with the AIS3 exporter.

Examples (from repo root):
  uv run python -m research.tinystories.train_vi --data-dir data/vi_tinystories/vocab-32768 --steps 3000 --tag vi-demo
  uv run python -m research.tinystories.train_vi --data-dir data/vi_tinystories/vocab-32768 --init runs/ple-something-s0.pt --steps 1500 --lr 0.0001 --tag vi-finetune

For --init, tokenizer SHA-256 must be EXACTLY the checkpoint's recorded SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from model import Config, TinyLM, make_model
from .train import Batcher, evaluate, get_device, lr_at

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--init", type=Path, help="Trusted PyTorch PLE .pt checkpoint (same tokenizer and shape)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--target-core", type=int, default=559000)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ple-dim", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="vi-demo")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise SystemExit("--tag must contain only letters, digits, underscores or hyphens.")
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise SystemExit("steps, batch-size, eval-every must be >= 1.")
    if args.eval_iters < 1 or args.warmup < 1:
        raise SystemExit("eval-iters and warmup must be >= 1.")

    token_path = args.data_dir / "tokenizer.json"
    bins = [args.data_dir / "train.bin", args.data_dir / "val.bin"]
    if not token_path.is_file() or not all(path.is_file() for path in bins):
        raise SystemExit(f"{args.data_dir}: missing tokenizer.json, train.bin or val.bin. Run prepare_vi first.")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(token_path))
    vocab_capacity = int(args.data_dir.name.removeprefix("vocab-"))
    if not 0 < tok.get_vocab_size() <= vocab_capacity <= 65536:
        raise SystemExit("Tokenizer vocabulary and data directory capacity are inconsistent.")
    tok_sha = hashlib.sha256(token_path.read_bytes()).hexdigest()
    if any(path.stat().st_size % 2 for path in bins):
        raise SystemExit("Token bins must contain uint16 values (file sizes must be even).")

    torch.manual_seed(args.seed)
    device = get_device()
    if args.init:
        # Never load arbitrary untrusted .pt files: torch.load can execute pickled objects.
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("tokenizer_sha256") != tok_sha:
            raise SystemExit(
                "Tokenizer SHA-256 differs from checkpoint; do not fine-tune with a new tokenizer. "
                "Reuse the original tokenizer or retrain from scratch."
            )
        cfg = Config(**ck["cfg"])
        if cfg.arm != "ple" or cfg.vocab_size != vocab_capacity:
            raise SystemExit("Checkpoint must be a PLE model with the selected vocab capacity.")
        model = TinyLM(cfg)
        model.load_state_dict(ck["state"], strict=True)
        print(f"Fine-tuning trusted checkpoint: {args.init}")
    else:
        cfg = Config(
            arm="ple", vocab_size=vocab_capacity, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads, ple_dim=args.ple_dim,
            seq_len=args.seq_len,
        )
        model = make_model("ple", args.target_core, cfg)
        cfg = model.cfg
        print("Training new Vietnamese tokenizer and model from scratch.")
    model = model.to(device)
    if any(path.stat().st_size // 2 <= cfg.seq_len + 2 for path in bins):
        raise SystemExit(f"Dataset too small for model seq_len={cfg.seq_len}. Add more Vietnamese stories.")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in name or "tok_emb" in name else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    train_batch = Batcher("train", args.batch_size, cfg.seq_len, device, args.data_dir, seed=args.seed)
    val_batch = Batcher("val", args.batch_size, cfg.seq_len, device, args.data_dir)
    name = f"ple-{args.tag}-s{args.seed}"
    out_pt, out_json = RUNS / f"{name}.pt", RUNS / f"{name}.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    if out_pt.exists() or out_json.exists():
        raise SystemExit(f"{name}: output exists. Choose another --tag to preserve previous checkpoints.")

    history = []
    start = time.monotonic()
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, min(args.warmup, args.steps))
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step + 1 == args.steps:
            score = evaluate(model, val_batch, args.eval_iters)
            rec = {"step": step + 1, "train_loss": float(loss.item()), "val_loss": float(score), "val_ppl": math.exp(score)}
            history.append(rec)
            print(f"step {step+1:5d} | train={rec['train_loss']:.4f} val={score:.4f} ppl={rec['val_ppl']:.2f}", flush=True)

    budget = model.param_budget()
    metadata = {
        "arm": "ple", "tag": args.tag, "seed": args.seed, "config": vars(cfg),
        "tokenizer_sha256": tok_sha, "init_checkpoint": str(args.init) if args.init else None,
        "training": {"steps": args.steps, "lr": args.lr, "batch_size": args.batch_size,
                     "eval_every": args.eval_every, "eval_iters": args.eval_iters, "warmup": args.warmup},
        "params": budget, "history": history,
        "final_val": history[-1]["val_loss"],
        "final_ppl": history[-1]["val_ppl"],
        "wall_seconds": round(time.monotonic() - start, 1),
    }
    out_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"cfg": vars(cfg), "state": model.cpu().state_dict(), "tokenizer_sha256": tok_sha,
         "seed": args.seed, "tag": args.tag, "name": name,
         "training": metadata["training"], "init_checkpoint": metadata["init_checkpoint"]},
        out_pt,
    )
    print(f"Saved {out_pt}, {out_json} | params={budget['total']:,}. Evaluate before export.")


if __name__ == "__main__":
    main()
