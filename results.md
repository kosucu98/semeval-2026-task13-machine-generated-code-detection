# Results

This file summarizes the results of my contributions to our SemEval 2026 Task 13 team submission.

My work focused on:

- **Subtask A:** binary classification of human-written vs. machine-generated code
- **Subtask C:** classification of human-written, machine-generated, hybrid, and adversarial code

## Evaluation Metric

The main evaluation metric was **Macro-F1**, which gives equal weight to each class and is suitable for classification tasks where class balance and per-class performance are important.

## Subtask A Results

| Model / Approach | Description | Validation Macro-F1 | Unseen Test Macro-F1 |
|---|---|---:|---:|
| LLaMA-2-7B sequence classification | Fine-tuned on 10K training samples | 0.99 | 0.38377 |
| CodeBERT baseline | Official task baseline provided by the organizers | Not reported | 0.30530 |

## Subtask C Results

| Model / Approach | Description | Validation Macro-F1 | Unseen Test Macro-F1 |
|---|---|---:|---:|
| LLaMA-2-7B sequence classification | Fine-tuned on 10K training samples | 0.74 | 0.56920 |
| CodeBERT baseline | Official task baseline provided by the organizers | Not reported | 0.48120 |

## Notes

- Results shown here are limited to the parts of the project related to my contribution, mainly Subtasks A and C.
- Subtask B was handled by my project partner, Osman Yigit Kandemir, and is therefore not included in this individual results summary.
- ## Notes

- Results shown here are limited to the parts of the project related to my contribution, mainly Subtasks A and C.
- Subtask B was handled by my project partner, Osman Yigit Kandemir, and is therefore not included in this individual results summary.
- The training and validation subsets were sampled separately from the official datasets using fixed random seeds of 42 and 43 respectively, with no overlap between the two sets.
- The models performed considerably better on the validation set than on the unseen test set, especially for Subtask A.
- The LLaMA-2-7B models were fine-tuned using Hugging Face Transformers and evaluated with F1-based classification metrics.
- Raw datasets, trained checkpoints, Hugging Face cache files, and Kaggle submission files are not included in this repository.
- The models performed considerably better on the validation set than on the unseen test set, especially for Subtask A.
- The LLaMA-2-7B models were fine-tuned using Hugging Face Transformers and evaluated with F1-based classification metrics.
- Raw datasets, trained checkpoints, Hugging Face cache files, and Kaggle submission files are not included in this repository.
