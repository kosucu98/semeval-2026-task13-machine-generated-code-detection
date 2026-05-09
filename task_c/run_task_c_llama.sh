#!/usr/bin/env bash
#SBATCH --job-name=subtask_c_llama
#SBATCH --partition=epyc-gpu
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=log.%x.%j.out

# Optional: uncomment and edit if your cluster requires an account
# #SBATCH --account=your-account-name

module purge
module load anaconda
conda activate your-conda-environment

echo "Loaded Anaconda environment"

# Offline / reproducibility settings
export WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

# Set cache directories for your environment
export HF_HOME=/path/to/your/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# Local LLaMA model directory
export LLAMA_DIR=/path/to/Llama-2-7b-hf

echo "Using LLaMA from: $LLAMA_DIR"
echo "HF_HOME: $HF_HOME"

echo "Python executable: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"

python -c "import importlib
try:
    accel = importlib.import_module('accelerate')
    print('Accelerate version:', accel.__version__)
except Exception:
    print('Accelerate not installed in this environment')
"

echo "=== GPU diagnostics ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

python - <<'PY'
import torch
import sys

print("python:", sys.executable)
print("torch:", getattr(torch, "__version__", "not-installed"))
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("torch_cuda_version:", getattr(torch.version, "cuda", "unknown"))
PY

# Run Task C LLaMA training with Accelerate
accelerate launch --num_processes 2 --mixed_precision bf16 train_task_c_llama.py \
  --train_parquet /path/to/task_c_training_set_1.parquet \
  --val_parquet /path/to/task_c_validation_set.parquet \
  --test_parquet /path/to/task_c_test.parquet \
  --model_dir "$LLAMA_DIR" \
  --output_dir taskC-llama-model \
  --max_length 256 \
  --batch_size 4 \
  --grad_accum 2 \
  --epochs 3 \
  --learning_rate 2e-5