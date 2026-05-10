#!/usr/bin/env python3

import os
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from itertools import chain
from tqdm import tqdm

# -------------------------------
# LIMITS FOR SMALL DEBUG RUNS
# Set to an integer or leave as None for full data
# -------------------------------
TRAIN_LIMIT = 10000
VAL_LIMIT   = 1000

# -------------------------------
# DEFAULT TEST + SUBMISSION OUTPUT
# -------------------------------
DEFAULT_TEST_PARQUET = "/hpc/gpfs2/scratch/u/lada58fu/task_c/test.parquet"
DEFAULT_SUBMISSION_CSV = "/hpc/gpfs2/scratch/u/lada58fu/task_c/submission_taskC_llama.csv"


def _is_ddp():
    # True when launched via accelerate/torchrun with distributed env vars
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ


def _is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


class LlamaTrainerTaskC:
    def __init__(
        self,
        max_length=256,
        model_dir="/hpc/gpfs2/scratch/g/coling-lehre/models/meta-llama/Llama-2-7b-hf",
    ):
        self.max_length = max_length
        self.model_dir = model_dir

        self.tokenizer = None
        self.model = None
        self.num_labels = None

        # for label remap: original_label(int) -> internal_id(int)
        self.label2id = None
        self.id2label = None

    def load_and_prepare_data(self, train_parquet, val_parquet):
        # Read train/val via pandas
        df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet)

        if "code" not in df.columns or "label" not in df.columns:
            raise ValueError("Train dataset must contain 'code' and 'label' columns")
        if "code" not in val_df.columns or "label" not in val_df.columns:
            raise ValueError("Val dataset must contain 'code' and 'label' columns")

        df = df.dropna(subset=["code", "label"]).copy()
        val_df = val_df.dropna(subset=["code", "label"]).copy()

        df["label"] = df["label"].astype(int)
        val_df["label"] = val_df["label"].astype(int)

        # --- Random limits ---
        if TRAIN_LIMIT is not None:
            n = min(int(TRAIN_LIMIT), len(df))
            print(f"[data] Using RANDOM {n} training samples (seed=42).")
            df = df.sample(n=n, random_state=42).reset_index(drop=True).copy()
        else:
            print(f"[data] Using ALL training samples: {len(df)}")

        if VAL_LIMIT is not None:
            n = min(int(VAL_LIMIT), len(val_df))
            print(f"[data] Using RANDOM {n} validation samples (seed=43).")
            val_df = val_df.sample(n=n, random_state=43).reset_index(drop=True).copy()
        else:
            print(f"[data] Using ALL validation samples: {len(val_df)}")

        # --- Label remap (safe for non-0..K-1 labels) ---
        uniq_train = sorted(df["label"].unique().tolist())
        self.label2id = {lab: i for i, lab in enumerate(uniq_train)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}
        self.num_labels = len(self.label2id)

        unseen = sorted(set(val_df["label"].unique().tolist()) - set(uniq_train))
        if unseen:
            raise ValueError(f"Validation has labels not in training: {unseen}")

        df["label"] = df["label"].map(self.label2id).astype(int)
        val_df["label"] = val_df["label"].map(self.label2id).astype(int)

        print("[data] unique train labels (original):", uniq_train)
        print("[data] label2id (original->internal):", self.label2id)
        print(f"[data] train={len(df)} val={len(val_df)} num_labels={self.num_labels}")

        return df, val_df

    def initialize_model_and_tokenizer(self):
        local_dir = os.environ.get("LLAMA_DIR", self.model_dir)
        if not os.path.isdir(local_dir):
            raise ValueError(f"Local LLaMA model directory not found: {local_dir}")

        print(f"[init] Loading tokenizer/model from: {local_dir} (local_only=True)")

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True)

        if self.tokenizer.pad_token is None:
            print("[init] Tokenizer has no pad_token. Setting pad_token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else (torch.float16 if torch.cuda.is_available() else torch.float32)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_dir,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            # store mapping in config (string keys are safer)
            label2id={str(k): v for k, v in self.label2id.items()},
            id2label={v: str(k) for k, v in self.label2id.items()},
        )

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        #if hasattr(self.model, "gradient_checkpointing_enable"):
            #self.model.gradient_checkpointing_enable()
            #print("[init] Gradient checkpointing enabled.")

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["code"],
            truncation=True,
            max_length=self.max_length,
        )

    def prepare_datasets(self, df, val_df):
        print("[data] Converting DataFrames to Datasets...")

        # CHANGED: disk caching to avoid duplicate tokenization work (and RAM spikes) under DDP
        cache_root = os.environ.get("HF_DATASETS_CACHE", "./tokenized_cache_taskc")
        os.makedirs(cache_root, exist_ok=True)
        train_cache_dir = os.path.join(cache_root, f"train_len{self.max_length}_seed42_n{len(df)}")
        val_cache_dir = os.path.join(cache_root, f"val_len{self.max_length}_seed43_n{len(val_df)}")

        if os.path.isdir(train_cache_dir) and os.path.isdir(val_cache_dir):
            print("[data] Loading tokenized datasets from disk cache...")
            train_dataset = Dataset.load_from_disk(train_cache_dir)
            val_dataset = Dataset.load_from_disk(val_cache_dir)
            return train_dataset, val_dataset

        # If distributed, only rank0 builds + saves, others wait and then load.
        if _is_ddp() and (not _is_main_process()):
            # wait for main process to finish writing cache
            torch.distributed.barrier()
            print("[data] Loading tokenized datasets from disk cache (after barrier)...")
            train_dataset = Dataset.load_from_disk(train_cache_dir)
            val_dataset = Dataset.load_from_disk(val_cache_dir)
            return train_dataset, val_dataset

        train_dataset = Dataset.from_pandas(df[["code", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["code", "label"]])

        print("[data] Tokenizing datasets...")
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=["code"])
        val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=["code"])

        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        print("[data] Saving tokenized datasets to disk cache...")
        train_dataset.save_to_disk(train_cache_dir)
        val_dataset.save_to_disk(val_cache_dir)

        if _is_ddp():
            torch.distributed.barrier()

        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        # supports tuple OR EvalPrediction
        try:
            predictions, labels = eval_pred
        except Exception:
            predictions, labels = eval_pred.predictions, eval_pred.label_ids

        preds = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}

    def train(self, train_dataset, val_dataset, output_dir="./results",
              num_epochs=8, batch_size=2, learning_rate=2e-5, grad_accum=8, num_workers=2):

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            weight_decay=0.01,

            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,

            remove_unused_columns=False,
            save_total_limit=2,
            report_to=[],

            bf16=bf16_ok,
            fp16=(not bf16_ok and torch.cuda.is_available()),

            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,

            seed=42,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("[train] Starting training...")
        trainer.train()

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[train] Saved model/tokenizer to {output_dir}")

        return trainer

    def evaluate_model(self, trainer, val_dataset):
        print("[eval] Evaluating on validation set...")
        results = trainer.evaluate(eval_dataset=val_dataset)
        print("[eval] Results:", results)

        pred = trainer.predict(val_dataset)
        y_pred = np.argmax(pred.predictions, axis=1)
        y_true = pred.label_ids
        print("[eval] Classification report (internal ids 0..K-1):")
        print(classification_report(y_true, y_pred, zero_division=0))

    @torch.no_grad()
    def predict_test_to_csv(self, parquet_path, output_path,
                            batch_size=16, device=None):
        """
        Streaming inference over parquet with columns ['ID','code'] and writes CSV: ID,label
        label = ORIGINAL label id (before remap)
        """
        # CHANGED: in DDP, only rank0 should write predictions
        if _is_ddp() and (not _is_main_process()):
            return

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self.model.to(device).eval()
        tokenizer = self.tokenizer

        ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
        it = iter(ds)
        first = next(it)

        if "ID" in first:
            id_key = "ID"
        elif "id" in first:
            id_key = "id"
        else:
            raise ValueError(f"Test parquet must contain 'ID' or 'id'. Keys: {list(first.keys())}")

        if "code" not in first:
            raise ValueError(f"Test parquet must contain 'code'. Keys: {list(first.keys())}")

        stream = chain([first], it)

        def batcher(iterator, bs):
            buf = []
            for ex in iterator:
                buf.append(ex)
                if len(buf) == bs:
                    yield buf
                    buf = []
            if buf:
                yield buf

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        with open(output_path, "w") as f:
            f.write("ID,label\n")

            for batch in tqdm(batcher(stream, batch_size), desc="Predicting"):
                codes = [row["code"] for row in batch]
                ids = [row[id_key] for row in batch]

                enc = tokenizer(
                    codes,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                if torch.cuda.is_available():
                    dtype = torch.bfloat16 if use_bf16 else torch.float16
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        logits = model(**enc).logits
                else:
                    logits = model(**enc).logits

                pred_internal = logits.argmax(dim=-1).detach().cpu().tolist()
                pred_original = [int(self.id2label[i]) for i in pred_internal]

                for ex_id, pred in zip(ids, pred_original):
                    f.write(f"{ex_id},{pred}\n")

        print(f"[predict] Predictions saved to {output_path}")

    def run_full_pipeline(self, train_parquet, val_parquet,
                          output_dir="./results", num_epochs=8,
                          batch_size=2, learning_rate=2e-5,
                          grad_accum=8, num_workers=2):
        print("[pipeline] Running full training pipeline with LLaMA (Task C)...")

        df, val_df = self.load_and_prepare_data(train_parquet, val_parquet)
        self.initialize_model_and_tokenizer()
        train_dataset, val_dataset = self.prepare_datasets(df, val_df)

        trainer = self.train(
            train_dataset, val_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            grad_accum=grad_accum,
            num_workers=num_workers
        )

        self.evaluate_model(trainer, val_dataset)
        print("[pipeline] Done.")
        return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train LLaMA model for SemEval Task C and optionally run inference."
    )

    parser.add_argument("--train_parquet", type=str,
                        default="/hpc/gpfs2/scratch/u/lada58fu/task_c/task_c_training_set_1.parquet")
    parser.add_argument("--val_parquet", type=str,
                        default="/hpc/gpfs2/scratch/u/lada58fu/task_c/task_c_validation_set.parquet")

    parser.add_argument("--output_dir", type=str, default="taskC-llama-model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)

    # add missing args so CLI actually works
    parser.add_argument("--model_dir", type=str,
                        default="/hpc/gpfs2/scratch/g/coling-lehre/models/meta-llama/Llama-2-7b-hf",
                        help="Local offline LLaMA HF directory (can also set env LLAMA_DIR).")

    # Prediction defaults are now INSIDE CODE (will run by default)
    parser.add_argument("--test_parquet", type=str, default=DEFAULT_TEST_PARQUET)
    parser.add_argument("--predictions_csv", type=str, default=DEFAULT_SUBMISSION_CSV)

    args = parser.parse_args()

    ##global TRAIN_LIMIT, VAL_LIMIT
    ##TRAIN_LIMIT = args.train_limit
    ##VAL_LIMIT = args.val_limit

    os.makedirs(args.output_dir, exist_ok=True)

    trainer_obj = LlamaTrainerTaskC(
        max_length=args.max_length,
        model_dir=args.model_dir,
    )

    trainer_obj.run_full_pipeline(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
    )

    # will run automatically because args.test_parquet has a default path
    if args.test_parquet is not None:
        print(f"[predict] Running prediction on test parquet: {args.test_parquet}")
        trainer_obj.predict_test_to_csv(
            parquet_path=args.test_parquet,
            output_path=args.predictions_csv,
            batch_size=max(16, args.batch_size),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("[predict] Wrote:", args.predictions_csv)


if __name__ == "__main__":
    # CHANGED: ensure torch.distributed is initialized before barriers when using accelerate/DDP
    if _is_ddp() and (not torch.distributed.is_initialized()):
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    main()
