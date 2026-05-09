import os
os.environ["WANDB_DISABLED"] = "true"
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
VAL_LIMIT = 1000


class LlamaTrainer:
    def __init__(self, max_length=512, model_dir=None):
        self.max_length = max_length
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.device = None

    def load_and_prepare_data(self, train_parquet, val_parquet):
        try:
            df = pd.read_parquet(train_parquet)

            print(f"Dataset columns: {df.columns.tolist()}")
            print(f"Sample data:\n{df.head()}")

            if "code" not in df.columns or "label" not in df.columns:
                raise ValueError("Training dataset must contain 'code' and 'label' columns")

            df = df.dropna(subset=["code", "label"]).copy()
            df["label"] = df["label"].astype(int)
            self.num_labels = df["label"].nunique()

            print("unique train labels:", sorted(df["label"].unique().tolist()))

            if TRAIN_LIMIT is not None:
                n = min(TRAIN_LIMIT, len(df))
                print(f"Using RANDOM {n} training samples (seed=42).")
                df = df.sample(n=n, random_state=42).reset_index(drop=True).copy()

            print(f"Number of unique labels: {self.num_labels}")
            print(f"Label range: {df['label'].min()} to {df['label'].max()}")
            print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

            val_df = pd.read_parquet(val_parquet)

            if "code" not in val_df.columns or "label" not in val_df.columns:
                raise ValueError("Validation dataset must contain 'code' and 'label' columns")

            val_df = val_df.dropna(subset=["code", "label"]).copy()
            val_df["label"] = val_df["label"].astype(int)

            print("unique val labels:", sorted(val_df["label"].unique().tolist()))

            if VAL_LIMIT is not None:
                n = min(VAL_LIMIT, len(val_df))
                print(f"Using RANDOM {n} validation samples (seed=43).")
                val_df = val_df.sample(n=n, random_state=43).reset_index(drop=True).copy()

            print(f"Train samples: {len(df)}, Validation samples: {len(val_df)}")

            return df, val_df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def initialize_model_and_tokenizer(self):
        local_dir = os.environ.get("LLAMA_DIR", self.model_dir)

        if local_dir is None:
            raise ValueError(
                "No model directory provided. Use --model_dir or set the LLAMA_DIR environment variable."
            )

        if not os.path.isdir(local_dir):
            raise ValueError(f"Local LLaMA model directory not found: {local_dir}")

        print(f"Initializing LLaMA tokenizer and model from: {local_dir} (local_only=True)...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            print("Tokenizer has no pad_token. Setting pad_token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_dir,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print(f"Using device: {self.device}")
        print(f"Model initialized with {self.num_labels} labels")

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["code"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def prepare_datasets(self, df, val_df):
        print("Converting DataFrames to Datasets...")

        train_dataset = Dataset.from_pandas(df[["code", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["code", "label"]])

        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["code"],
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["code"],
        )

        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir="taskA-llama-model",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
    ):
        print("Starting training...")
        print(self.model)
        print(self.model.device)

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=2,
            report_to=[],
            bf16=bf16_ok,
            fp16=False,
            gradient_accumulation_steps=4,
            dataloader_num_workers=1,
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

        print("Start training")
        trainer.train()

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Training completed. Model saved to {output_dir}")

        return trainer

    def evaluate_model(self, trainer, val_dataset):
        print("Evaluating model on validation set...")
        results = trainer.evaluate(eval_dataset=val_dataset)
        print("Evaluation results:", results)

        val_predictions = trainer.predict(val_dataset)
        preds = np.argmax(val_predictions.predictions, axis=1)
        labels = val_predictions.label_ids
        print("Classification Report:\n", classification_report(labels, preds, zero_division=0))

    def run_full_pipeline(
        self,
        train_parquet,
        val_parquet,
        output_dir="taskA-llama-model",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
    ):
        print("Running full training pipeline with LLaMA...")

        try:
            df, val_df = self.load_and_prepare_data(train_parquet, val_parquet)

            self.initialize_model_and_tokenizer()

            train_dataset, val_dataset = self.prepare_datasets(df, val_df)

            trainer = self.train(
                train_dataset,
                val_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            self.evaluate_model(trainer, val_dataset)

            print("Pipeline completed successfully!")
            return trainer

        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise


@torch.no_grad()
def predict_with_trainer(trainer_obj, parquet_path, output_path, max_length=512, batch_size=16, device=None):
    """
    Uses trainer_obj.model and trainer_obj.tokenizer to run streaming inference
    over a parquet file with columns ['ID', 'code'] and writes submission CSV: ID,label
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = trainer_obj.model
    tokenizer = trainer_obj.tokenizer if hasattr(trainer_obj, "tokenizer") else None

    if tokenizer is None:
        raise ValueError("trainer_obj must have a tokenizer.")

    model.to(device)
    model.eval()

    ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)

    it = iter(ds)
    first = next(it)

    if not {"ID", "code"}.issubset(first.keys()):
        raise ValueError("Parquet file must contain 'ID' and 'code' columns")

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

    with open(output_path, "w") as f:
        f.write("ID,label\n")

        for batch in tqdm(batcher(stream, batch_size), desc="Predicting"):
            codes = [row["code"] for row in batch]
            ids = [row["ID"] for row in batch]

            enc = tokenizer(
                codes,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_labels = logits.argmax(dim=-1).cpu().tolist()

            for ex_id, pred in zip(ids, pred_labels):
                f.write(f"{ex_id},{pred}\n")

    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train LLaMA model for SemEval Task A and optionally run inference."
    )

    parser.add_argument("--train_parquet", type=str, required=True, help="Path to Task A training parquet file.")
    parser.add_argument("--val_parquet", type=str, required=True, help="Path to Task A validation parquet file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Local LLaMA model directory.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="taskA-llama-model",
        help="Directory where the fine-tuned model will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size used for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization.")

    parser.add_argument(
        "--test_parquet",
        type=str,
        default=None,
        help="Optional test parquet for inference.",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default="submission_llama.csv",
        help="Where to write prediction CSV if --test_parquet is provided.",
    )

    args = parser.parse_args()

    trainer_obj = LlamaTrainer(
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
    )

    if args.test_parquet is not None:
        print(f"Running prediction on test parquet: {args.test_parquet}")
        predict_with_trainer(
            trainer_obj=trainer_obj,
            parquet_path=args.test_parquet,
            output_path=args.predictions_csv,
            max_length=args.max_length,
            batch_size=16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Wrote submission CSV:", args.predictions_csv)


if __name__ == "__main__":
    main()