"""
Quick evaluation of the fine-tuned model on val set.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --adapter outputs/checkpoints/final --n 50
"""

import argparse
import json
import sys
from pathlib import Path

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

import torch
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocess import extract_answer


def load_val(path: str, n: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:n] if n > 0 else rows


def run_eval(adapter_path: str, val_path: str, n: int, max_new_tokens: int) -> None:
    print(f"Loading base model + adapter from {adapter_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # enable faster inference kernel

    rows = load_val(val_path, n)
    print(f"Evaluating on {len(rows)} examples...\n")

    correct = 0
    for i, row in enumerate(rows):
        prompt_text = (
            f"### Problem:\n{row['prompt']}\n\n"
            "### Reasoning:\n"
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        predicted = extract_answer(generated)
        ground_truth = row["answer"].strip()
        is_correct = (
            predicted is not None
            and predicted.lower().strip() == ground_truth.lower().strip()
        )
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"[{i+1:>3}/{len(rows)}] {status}  pred={repr(predicted):<20} gt={repr(ground_truth)}")
        if predicted is None:
            print(f"         RAW: {repr(generated[:200])}")

    acc = correct / len(rows) * 100
    print(f"\n{'='*50}")
    print(f"Accuracy:  {correct}/{len(rows)}  ({acc:.1f}%)")
    print(f"Adapter:   {adapter_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",        default="outputs/checkpoints/final")
    parser.add_argument("--val",            default="data/splits/val_cot.jsonl")
    parser.add_argument("--n",    type=int, default=50,  help="examples to eval (0=all)")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()
    run_eval(args.adapter, args.val, args.n, args.max_new_tokens)


if __name__ == "__main__":
    main()
