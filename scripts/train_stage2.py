"""
Stage 2 — GRPO self-improvement on all 9.5k examples.

Loads the Stage 1 CoT checkpoint and runs GRPO:
  - Model generates reasoning + answer autoregressively
  - Reward = 1.0 if final answer matches ground truth, else 0.0
  - No CoT rationales needed — only (prompt, answer) pairs

Usage:
    python scripts/train_stage2.py
"""

import os
import sys

# Disable torch.compile / dynamo before any other imports
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import torch
torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else lambda f: f)
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.reset()

import csv
from pathlib import Path

import yaml
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocess import extract_answer


MAX_PROMPT_CHARS = 800  # ~256 tokens — truncate long prompts before they reach the model

def load_train_csv(csv_path: str) -> HFDataset:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = []
    for row in rows:
        prompt = f"### Problem:\n{row['prompt']}\n\n### Reasoning:\n"
        records.append({
            "prompt":       prompt[:MAX_PROMPT_CHARS],
            "ground_truth": row["answer"].strip(),
        })
    return HFDataset.from_list(records)


def make_reward_fn(dataset: HFDataset):
    ground_truths = dataset["ground_truth"]

    def reward_fn(completions, prompts, **kwargs):
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

    adapter_repo = "suriya-mars/qwen2.5-3b-wonderland-stage1"
    print(f"Loading base model + Stage 1 adapter from {adapter_repo} ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_repo, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\nLoading 9.5k train examples ...")
    train_ds = load_train_csv("data/raw/train.csv")
    print(f"  rows={len(train_ds)}  columns={train_ds.column_names}")

    reward_fn = make_reward_fn(train_ds)

    grpo_config = GRPOConfig(
        output_dir=stage2_dir,

        per_device_train_batch_size=4,
        num_generations=8,
        max_completion_length=512,
        gradient_accumulation_steps=4,
        beta=0.01,

        num_train_epochs=1,
        learning_rate=5e-6,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_grad_norm=0.1,

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
