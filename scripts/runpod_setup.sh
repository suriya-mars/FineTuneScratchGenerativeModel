#!/bin/bash
set -e

# Authenticate with HuggingFace
hf login --token "$HF_TOKEN"

rm -rf /workspace/project
git clone https://github.com/suriya-mars/FineTuneScratchGenerativeModel.git /workspace/project
cd /workspace/project

pip install -q unsloth trl peft datasets transformers accelerate bitsandbytes huggingface_hub

hf download suriya-mars/wonderland-data train.csv --local-dir data/raw --repo-type dataset

mkdir -p outputs/checkpoints/stage2

PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train_stage2.py 2>&1 | tee /workspace/stage2_log.txt

# Upload only if training completed successfully
if [ -d "outputs/checkpoints/stage2/final" ]; then
    echo "Uploading Stage 2 adapter to HuggingFace..."
    hf upload suriya-mars/qwen2.5-3b-wonderland-stage2 outputs/checkpoints/stage2/final --repo-type model
    echo "TRAINING_DONE"
else
    echo "ERROR: outputs/checkpoints/stage2/final not found — training may have failed"
    cat /workspace/stage2_log.txt
    exit 1
fi
