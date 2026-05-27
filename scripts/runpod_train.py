"""
Launch Stage 2 GRPO training on RunPod via SDK.

Usage:
    export RUNPOD_API_KEY=your_key_here
    python scripts/runpod_train.py
"""

import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download
import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]

GITHUB_REPO   = "https://github.com/suriya-mars/FineTuneScratchGenerativeModel.git"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")   # needed to pull models if private

SETUP_COMMANDS = f"""
set -e
git clone {GITHUB_REPO} /workspace/project
cd /workspace/project

pip install -q unsloth trl peft datasets transformers accelerate bitsandbytes huggingface_hub

# Download train.csv from HuggingFace
huggingface-cli download suriya-mars/wonderland-data train.csv --local-dir data/raw

PYTORCH_ALLOC_CONF=expandable_segments:True \\
    python scripts/train_stage2.py 2>&1 | tee /workspace/stage2_log.txt

# Upload Stage 2 adapter back to HuggingFace when done
huggingface-cli upload suriya-mars/qwen2.5-3b-wonderland-stage2 outputs/checkpoints/stage2/final .

echo "TRAINING_DONE"
"""


def launch_pod() -> str:
    pod = runpod.create_pod(
        name="grpo-stage2",
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        gpu_type_id="NVIDIA GeForce RTX 3090",   # 24GB VRAM
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=20,
        container_disk_in_gb=20,
        docker_args=f'bash -c "{SETUP_COMMANDS}"',
    )
    pod_id = pod["id"]
    print(f"Pod launched: {pod_id}")
    return pod_id


def wait_for_completion(pod_id: str) -> None:
    print("Waiting for training to complete...")
    while True:
        status = runpod.get_pod(pod_id)
        state  = status.get("desiredStatus", "UNKNOWN")
        print(f"  Status: {state}")
        if state in ("EXITED", "TERMINATED"):
            break
        time.sleep(60)


def terminate_pod(pod_id: str) -> None:
    runpod.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated.")


def download_adapter(local_dir: str = "outputs/checkpoints/stage2/final") -> None:
    print("Downloading Stage 2 adapter from HuggingFace...")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="suriya-mars/qwen2.5-3b-wonderland-stage2",
        local_dir=local_dir,
    )
    print(f"Saved to {local_dir}")


if __name__ == "__main__":
    pod_id = launch_pod()
    try:
        wait_for_completion(pod_id)
        download_adapter()
    finally:
        terminate_pod(pod_id)
