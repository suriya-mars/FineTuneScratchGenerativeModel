#!/bin/bash
set -e

pip install -q huggingface_hub

# Authenticate with HuggingFace
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

rm -rf /workspace/project
git clone https://github.com/suriya-mars/FineTuneScratchGenerativeModel.git /workspace/project
cd /workspace/project

pip install -q -r requirements_stage2.txt

# Download train.csv
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('data/raw', exist_ok=True)
path = hf_hub_download(repo_id='suriya-mars/wonderland-data', filename='train.csv', repo_type='dataset')
shutil.copy(path, 'data/raw/train.csv')
print('Downloaded train.csv')
"

mkdir -p outputs/checkpoints/stage2

# Create HF repo upfront so upload succeeds after training
python -c "
from huggingface_hub import HfApi
HfApi().create_repo('suriya-mars/qwen2.5-3b-wonderland-stage2', repo_type='model', exist_ok=True)
print('HF repo ready')
"

TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
UNSLOTH_COMPILE_DISABLE=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
    python scripts/train_stage2.py 2>&1 | tee /workspace/stage2_log.txt

# Upload only if training completed successfully
if [ -d "outputs/checkpoints/stage2/final" ]; then
    echo "Uploading Stage 2 adapter to HuggingFace..."
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('suriya-mars/qwen2.5-3b-wonderland-stage2', repo_type='model', exist_ok=True)
api.upload_folder(folder_path='outputs/checkpoints/stage2/final', repo_id='suriya-mars/qwen2.5-3b-wonderland-stage2', repo_type='model')
print('Upload complete')
"
    echo "TRAINING_DONE"
else
    echo "ERROR: outputs/checkpoints/stage2/final not found — training may have failed"
    tail -50 /workspace/stage2_log.txt
    exit 1
fi
