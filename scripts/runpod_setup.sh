#!/bin/bash
set -eo pipefail

pip install -q huggingface_hub

# Authenticate with HuggingFace
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

rm -rf /workspace/project
git clone https://github.com/suriya-mars/FineTuneScratchGenerativeModel.git /workspace/project
cd /workspace/project

pip install -q -r requirements_stage2.txt
python -c "import transformers, torch, trl; print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'trl:', trl.__version__)"

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

TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
UNSLOTH_COMPILE_DISABLE=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
    python scripts/train_stage2.py 2>&1 | tee /workspace/stage2_log.txt

echo "TRAINING_DONE"
