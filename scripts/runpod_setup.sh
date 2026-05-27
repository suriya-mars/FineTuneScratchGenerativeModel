#!/bin/bash
set -e

git clone https://github.com/suriya-mars/FineTuneScratchGenerativeModel.git /workspace/project
cd /workspace/project

pip install -q unsloth trl peft datasets transformers accelerate bitsandbytes huggingface_hub

huggingface-cli download suriya-mars/wonderland-data train.csv --local-dir data/raw --repo-type dataset

mkdir -p outputs/checkpoints/stage2

PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_stage2.py 2>&1 | tee /workspace/stage2_log.txt

huggingface-cli upload suriya-mars/qwen2.5-3b-wonderland-stage2 outputs/checkpoints/stage2/final .

echo "TRAINING_DONE"
