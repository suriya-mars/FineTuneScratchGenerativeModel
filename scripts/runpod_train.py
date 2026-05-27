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

SETUP_CMD = (
    "bash -c 'curl -fsSL https://raw.githubusercontent.com"
    "/suriya-mars/FineTuneScratchGenerativeModel/master/scripts/runpod_setup.sh | bash'"
)




def launch_pod() -> str:
    pod = runpod.create_pod(
        name="grpo-stage2",
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        gpu_type_id="NVIDIA GeForce RTX 3090",   # 24GB VRAM
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=20,
        container_disk_in_gb=20,
        docker_args=SETUP_CMD,
        env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
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
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=your_token")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise SystemExit("ERROR: RUNPOD_API_KEY is not set.")
    pod_id = launch_pod()
    try:
        wait_for_completion(pod_id)
        download_adapter()
    finally:
        terminate_pod(pod_id)
