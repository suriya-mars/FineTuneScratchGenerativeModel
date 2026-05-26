"""Dataset loading for CoT fine-tuning."""
import json
from pathlib import Path
from torch.utils.data import Dataset
import datasets
from datasets import Dataset as HFDataset


def load_jsonl_as_hf_dataset(path: str) -> HFDataset:
    """Load JSONL as a prompt-completion HuggingFace Dataset for SFTTrainer.

    Splits training_text at '### Reasoning:' so SFTConfig(completion_only_loss=True)
    masks the problem block and trains only on reasoning + answer.
    """
    split_marker = "### Reasoning:"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row["training_text"]
            idx = text.find(split_marker)
            if idx != -1:
                records.append({
                    "prompt":     text[:idx],
                    "completion": text[idx:],
                })
            else:
                records.append({"prompt": "", "completion": text})
    return HFDataset.from_list(records)


class CoTDataset(Dataset):
    """PyTorch Dataset with manual loss masking — alternative to SFTTrainer."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["training_text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Mask "### Problem:" section from loss — only train on reasoning + answer
        labels = list(input_ids)
        reasoning_ids = self.tokenizer.encode("### Reasoning:", add_special_tokens=False)
        prompt_end = _find_subsequence(input_ids, reasoning_ids)
        if prompt_end != -1:
            for i in range(prompt_end):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
    """Return start index of first occurrence of subseq in seq, or -1."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1
