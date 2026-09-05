"""T1 TAPT: continued MLM pretraining on organizer-provided input text.

Registered CREATIVE_HEADROOM_RESEARCH.md (commit cd77061) BEFORE any
training. Corpus = 612 train_task_1 + 217 dev_in paragraph texts (input
text only, no labels). Transductive: OOF-measurement-only pending a
written organizer ruling; nothing built on this checkpoint deploys.

Recipe (frozen): camelbert-msa-quarter @ 3e48534, dynamic masking 0.15,
max_length 512 / stride 128 windows, 100 epochs, batch 16, lr 5e-5
linear decay with 6% warmup, weight decay 0.01, grad clip 1.0,
seed 20260710, fp32 CPU, single seed (Gururangan variance caveat).

Example (from shared-task/):
  uv run scripts/tapt_pretrain.py
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import atomic_write_json, create_experiment_dir
from daleel.data import DEV_INPUT, TRAIN_TASK1, load_records
from daleel.runtime import EXPERIMENTS_DIR

from transformers import (  # noqa: E402 — after torch determinism setup below
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

BASE_REPO = "CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter"
BASE_REVISION = "3e48534705c153737cbec1c5748bb02359b7b239"
CHECKPOINT_DIR = (Path(__file__).resolve().parents[1] / "models"
                  / "camelbert-msa-quarter-tapt-transductive-v1")

SEED = 20260710
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
MLM_PROBABILITY = 0.15
MAX_LENGTH = 512
STRIDE = 128
PAD_TO_MULTIPLE_OF = 8


class WindowDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.special_tokens_mask = encodings["special_tokens_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "special_tokens_mask": self.special_tokens_mask[index],
        }


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    seed_everything(SEED)
    torch.set_num_threads(os.cpu_count() or 1)

    train = load_records(TRAIN_TASK1)
    dev = load_records(DEV_INPUT)
    texts = {r["paragraph_id"]: r["text"] for r in train}
    texts.update({r["paragraph_id"]: r["text"] for r in dev})
    corpus = [texts[pid] for pid in sorted(texts)]
    corpus_chars = sum(len(t) for t in corpus)
    if len(corpus) != 829 or corpus_chars != 340743:
        raise SystemExit(
            f"corpus drifted from registration: {len(corpus)} paragraphs, "
            f"{corpus_chars} chars (expected 829 / 340743)")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_REPO, revision=BASE_REVISION, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        BASE_REPO, revision=BASE_REVISION, attn_implementation="eager")
    model.train()

    encodings = tokenizer(
        corpus,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
    )
    dataset = WindowDataset(encodings)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROBABILITY,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )

    total_steps = EPOCHS * len(loader)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith(".bias") or "LayerNorm" in name
         else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    run_contract = {
        "script": "tapt_pretrain.py",
        "registration": "CREATIVE_HEADROOM_RESEARCH.md TAPT (commit cd77061)",
        "base": {"repository": BASE_REPO, "revision": BASE_REVISION},
        "corpus": {"paragraphs": len(corpus), "chars": corpus_chars,
                   "windows": len(dataset),
                   "sources": ["train_task_1.jsonl", "dev_in.jsonl"],
                   "transductive": True, "labels_used": False},
        "hyperparameters": {
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO, "max_grad_norm": MAX_GRAD_NORM,
            "mlm_probability": MLM_PROBABILITY, "max_length": MAX_LENGTH,
            "stride": STRIDE, "pad_to_multiple_of": PAD_TO_MULTIPLE_OF,
            "device": "cpu", "precision": "fp32",
        },
        "checkpoint_dir": str(CHECKPOINT_DIR),
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"tapt-mlm-camelbert-quarter-transductive-{len(corpus)}",
        run_contract)
    atomic_write_json(exp_dir / "config.json", run_contract)
    print(f"run dir: {exp_dir}", flush=True)
    print(f"windows: {len(dataset)}  steps/epoch: {len(loader)}  "
          f"total steps: {total_steps}", flush=True)

    epoch_losses = []
    started = time.time()
    for epoch in range(EPOCHS):
        running, batches = 0.0, 0
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += float(outputs.loss)
            batches += 1
        epoch_losses.append(round(running / batches, 4))
        elapsed = time.time() - started
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{EPOCHS}  mlm_loss {epoch_losses[-1]}  "
                  f"elapsed {elapsed / 60:.1f}m", flush=True)
            atomic_write_json(exp_dir / "metrics.json", {
                "status": "training", "epochs_done": epoch + 1,
                "epoch_mlm_loss": epoch_losses,
                "elapsed_seconds": round(elapsed, 1)})

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    weights_hash = subprocess.run(
        ["git", "hash-object", str(CHECKPOINT_DIR / "model.safetensors")],
        capture_output=True, text=True, check=True).stdout.strip()

    metrics = {
        "status": "complete",
        "epoch_mlm_loss": epoch_losses,
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "elapsed_seconds": round(time.time() - started, 1),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "weights_git_hash_object": weights_hash,
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    atomic_write_json(exp_dir / "provenance.json", {
        "derivation": "apache-2.0 base checkpoint + organizer-provided input "
                      "text only (no labels, no external data)",
        "legality": "transductive (includes dev_in.jsonl inputs); "
                    "OOF-measurement-only pending written organizer ruling; "
                    "must not deploy",
        "base": {"repository": BASE_REPO, "revision": BASE_REVISION,
                 "license": "apache-2.0"},
        "weights_git_hash_object": weights_hash,
    })
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"\nregistry revision (git hash-object): {weights_hash}", flush=True)


if __name__ == "__main__":
    main()
