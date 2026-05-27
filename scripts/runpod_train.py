"""
Launch Stage 2 GRPO training on RunPod via SDK.

Usage:
    export RUNPOD_API_KEY=your_key_here
    python scripts/runpod_train.py
"""

import os
import time
import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]

GITHUB_REPO   = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")   # needed to pull models if private

SETUP_COMMANDS = f"""
set -e
git clone {GITHUB_REPO} /workspace/project
cd /workspace/project

pip install -q unsloth trl peft datasets transformers accelerate bitsandbytes

# Copy Stage 1 adapter if you uploaded it to RunPod volume
# cp /runpod-volume/final/* outputs/checkpoints/final/

PYTORCH_ALLOC_CONF=expandable_segments:True \\
    python scripts/train_stage2.py 2>&1 | tee /workspace/stage2_log.txt

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


if __name__ == "__main__":
    pod_id = launch_pod()
    try:
        wait_for_completion(pod_id)
        print("\nDone! Download outputs/checkpoints/stage2/ from the pod volume.")
    finally:
        terminate_pod(pod_id)
