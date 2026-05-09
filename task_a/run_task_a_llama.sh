#!/usr/bin/env bash
#SBATCH --job-name=subtask_a_llama
#SBATCH --partition=epyc-gpu
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=log.%x.%j.out

# Optional: uncomment and edit if your cluster requires an account
# #SBATCH --account=your-account-name

module purge
module load anaconda
conda activate your-conda-environment

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

echo "Python executable: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"

echo "=== GPU diagnostics ==="
nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY

python -u train_task_a_llama.py \
  --train_parquet /path/to/task_a_training_set_1.parquet \
  --val_parquet /path/to/task_a_validation_set.parquet \
  --test_parquet /path/to/task_a_test.parquet \
  --model_dir "$LLAMA_DIR" \
  --output_dir taskA-llama-model \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --max_length 256