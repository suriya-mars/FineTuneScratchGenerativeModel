"""
SFT fine-tuning with Unsloth + TRL SFTTrainer.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/model_config.yaml
"""

# ── Python 3.14 / datasets compatibility shim ────────────────────────────────
# Python 3.14 made pickle._Pickler._batch_setitems(self, items, obj) required.
# dill and datasets.utils._dill both call it with only (self, items), crashing
# during Dataset fingerprinting.  Two patches needed after datasets is loaded:
#   1. Make pickle._Pickler._batch_setitems accept obj=None
#   2. Make datasets.utils._dill.Pickler._batch_setitems accept and forward obj
import unsloth          # noqa — imports datasets/dill; patches must follow

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

import argparse
import sys
from pathlib import Path

import yaml
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.model import load_model_and_tokenizer
from src.data.dataset import load_jsonl_as_hf_dataset


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    t_cfg    = config["training"]
    log_cfg  = config["logging"]
    data_cfg = config["data"]

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    print("\nLoading datasets...")
    train_ds = load_jsonl_as_hf_dataset(data_cfg["train_file"])
    val_ds   = load_jsonl_as_hf_dataset(data_cfg["val_file"])
    print(f"  train={len(train_ds)}  val={len(val_ds)}")
    print(f"  columns: {train_ds.column_names}")

    sft_config = SFTConfig(
        # ── output ──────────────────────────────────────────────────────
        output_dir=t_cfg["output_dir"],

        # ── optimisation ────────────────────────────────────────────────
        per_device_train_batch_size=t_cfg["batch_size"],
        gradient_accumulation_steps=t_cfg["grad_accum_steps"],
        num_train_epochs=t_cfg["num_epochs"],
        learning_rate=t_cfg["learning_rate"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",          # 8-bit AdamW saves ~1 GB vs fp32
        lr_scheduler_type="cosine",

        # ── sequence / loss ─────────────────────────────────────────────
        max_length=config["model"]["max_seq_len"],
        completion_only_loss=True,   # mask "### Problem:" block from loss

        # ── eval / save ─────────────────────────────────────────────────
        eval_strategy="steps",
        eval_steps=log_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=log_cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── misc ────────────────────────────────────────────────────────
        logging_steps=10,
        dataset_num_proc=1,
        seed=t_cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    print("\nStarting training...")
    trainer.train()

    final_dir = Path(t_cfg["output_dir"]) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved final model → {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT fine-tuning with Unsloth + TRL")
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
