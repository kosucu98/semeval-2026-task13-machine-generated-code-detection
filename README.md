# SemEval 2026 Task 13: Machine-Generated Code Detection

This repository contains my contributions to our team submission for **SemEval 2026 Task 13**, focused on detecting and attributing machine-generated code.

My work focused on **Subtask A** and **Subtask C**, where I implemented LLaMA-based sequence classification models.

## Competition and Dataset Information

More information about the SemEval 2026 Task 13 challenge, including the task description, subtasks, and datasets, can be found on the Kaggle page:

https://www.kaggle.com/datasets/daniilor/semeval-2026-task13

## Subtasks

- **Subtask A:** Binary classification — human-written vs. machine-generated code
- **Subtask B:** Multi-class attribution — handled by project partner Osman Yigit Kandemir
- **Subtask C:** Classification of human-written, machine-generated, hybrid, and adversarial code

## My Contributions

- Implemented LLaMA-2-7B fine-tuning pipelines for Subtask A and Subtask C
- Built training and validation workflows using PyTorch and Hugging Face Transformers
- Processed Parquet datasets with pandas and Hugging Face Datasets
- Added evaluation with accuracy, precision, recall, F1, and classification reports
- Implemented test-set inference and CSV generation for submission format
- Adapted scripts for the University of Augsburg LICCA HPC environment with local model paths and offline Hugging Face settings.

- ## Results

The models were evaluated using standard classification metrics, with particular focus on F1-score due to the classification setting.

| Subtask | Model / Approach | Training Setup | Macro-F1 |
|---|---|---:|---:|
| Subtask A | LLaMA-2-7B sequence classification | 10K training samples | 0.38377 |
| Subtask C | LLaMA-2-7B sequence classification | 10K training samples | 0.56920 |

More detailed results and notes are available in [`results_summary.md`](results_summary.md).

## Repository Structure

Each subtask folder contains the Python training/evaluation script and a shell script showing how the experiment was run on the Slurm-based HPC environment.

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── task_a/
│   ├── task_a_llama.py
│   └── run_task_a_llama.sh
└── task_c/
    ├── task_c_llama.py
    └── run_task_c_llama.sh
