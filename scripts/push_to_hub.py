"""
Push Stage 1 LoRA adapter to HuggingFace Hub.

Usage:
    huggingface-cli login          # one-time login
    python scripts/push_to_hub.py
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_USERNAME  = "suriya-mars"
REPO_NAME    = "qwen2.5-3b-wonderland-stage1"
LOCAL_FOLDER = "outputs/checkpoints/final"

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Repo: https://huggingface.co/{repo_id}")

print(f"Uploading {LOCAL_FOLDER} ...")
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=repo_id,
    repo_type="model",
)

print(f"\nDone! Adapter available at: https://huggingface.co/{repo_id}")
print(f"\nTo load on RunPod:")
print(f'  model = PeftModel.from_pretrained(base_model, "{repo_id}")')
