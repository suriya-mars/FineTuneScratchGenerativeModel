"""
Stage 2 — GRPO self-improvement on all 9.5k examples.

Loads the Stage 1 CoT checkpoint and runs GRPO:
  - Model generates reasoning + answer autoregressively
  - Reward = 1.0 if final answer matches ground truth, else 0.0
  - No CoT rationales needed — only (prompt, answer) pairs

This closes the train/inference gap from Stage 1's teacher forcing.

Usage:
    python scripts/train_stage2.py
"""

# ── Python 3.14 / datasets compatibility shim ────────────────────────────────
import unsloth
import pickle as _pickle
if hasattr(_pickle, "_Pickler"):
    _orig_base = _pickle._Pickler._batch_setitems
    def _base_batch(self, items, obj=None):
        _orig_base(self, items, obj if obj is not None else {})
    _pickle._Pickler._batch_setitems = _base_batch
try:
    import datasets.utils._dill as _dd
    _orig_dd = _dd.Pickler._batch_setitems
    def _dd_batch(self, items, obj=None):
        _orig_dd(self, items)
    _dd.Pickler._batch_setitems = _dd_batch
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"   # skip torch.compile — saves 30-60min startup

import csv
import sys
from pathlib import Path

import yaml
from datasets import Dataset as HFDataset
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocess import extract_answer


def load_train_csv(csv_path: str) -> HFDataset:
    """Load train.csv as a GRPO dataset — just prompt + ground_truth."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        records.append({
            "prompt":       f"### Problem:\n{row['prompt']}\n\n### Reasoning:\n",
            "ground_truth": row["answer"].strip(),
        })
    return HFDataset.from_list(records)


def make_reward_fn(dataset: HFDataset):
    """Return a reward function that looks up ground truth by index."""
    ground_truths = dataset["ground_truth"]

    def reward_fn(completions, prompts, **kwargs):
        # completions: list of generated strings (reasoning + answer)
        # prompts: list of input prompts — use index to match ground truth
        rewards = []
        for i, completion in enumerate(completions):
            gt = ground_truths[i % len(ground_truths)]
            predicted = extract_answer(completion)
            if predicted and predicted.lower().strip() == gt.lower().strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def main() -> None:
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)

    stage2_dir = config["training"]["output_dir"] + "/stage2"

    print("Loading base model + Stage 1 adapter from suriya-mars/qwen2.5-3b-wonderland-stage1 ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-bnb-4bit",
        max_seq_length=config["model"]["max_seq_len"],
        dtype=None,
        load_in_4bit=True,
    )

    # Load existing LoRA adapter as trainable — starts from Stage 1 weights
    model = PeftModel.from_pretrained(model, "suriya-mars/qwen2.5-3b-wonderland-stage1", is_trainable=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\nLoading 9.5k train examples ...")
    train_ds = load_train_csv("data/raw/train.csv")
    print(f"  rows={len(train_ds)}  columns={train_ds.column_names}")

    reward_fn = make_reward_fn(train_ds)

    grpo_config = GRPOConfig(
        output_dir=stage2_dir,

        # ── RTX 3090 24GB settings ───────────────────────────────────────
        per_device_train_batch_size=4,   # was 1
        num_generations=8,               # was 2 — more rollouts = better reward signal
        max_prompt_length=256,           # was 128
        max_completion_length=512,       # was 128 — full reasoning chains allowed
        gradient_accumulation_steps=4,   # was 8 — effective batch = 4*4 = 16 prompts
        beta=0.01,                       # was 0.0 — small KL keeps model from drifting too far

        # ── optimisation ────────────────────────────────────────────────
        num_train_epochs=1,
        learning_rate=5e-6,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_grad_norm=0.1,

        # ── logging / saving ────────────────────────────────────────────
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=42,
        torch_compile=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        reward_funcs=reward_fn,
        args=grpo_config,
    )

    print("\nStarting Stage 2 GRPO training...")
    trainer.train()

    final_dir = Path(stage2_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved Stage 2 model → {final_dir}")


if __name__ == "__main__":
    main()
