"""Training loop for LoRA fine-tuning.

Uses HuggingFace Trainer with configurable hyperparameters.
Logs metrics to CSV for later visualization.
Per-run persistence via JSON + CSV for crash-safe recovery.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


@dataclass
class TrainConfig:
    """Training hyperparameters for a single run."""

    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 512
    output_dir: str = "results/checkpoints"
    run_name: str = ""


@dataclass
class TrainResult:
    """Results from a single training run."""

    run_name: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    num_epochs: int
    trainable_params: int
    trainable_pct: float
    train_loss: float
    eval_loss: float
    train_time_seconds: float
    train_samples_per_second: float
    log_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return all fields except log_history (for JSON/CSV serialization)."""
        return {
            "run_name": self.run_name,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "trainable_params": self.trainable_params,
            "trainable_pct": self.trainable_pct,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "train_time_seconds": self.train_time_seconds,
            "train_samples_per_second": self.train_samples_per_second,
        }


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: TrainConfig,
    param_info: dict,
) -> TrainResult:
    """Run a single training experiment.

    Args:
        model: PEFT model with LoRA adapters.
        tokenizer: Tokenizer.
        train_dataset: Tokenized training dataset.
        eval_dataset: Tokenized validation dataset.
        config: Training hyperparameters.
        param_info: Dict from count_parameters().

    Returns:
        TrainResult with metrics.
    """
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    start = time.time()
    train_output = trainer.train()
    elapsed = time.time() - start

    eval_output = trainer.evaluate()

    result = TrainResult(
        run_name=config.run_name,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        trainable_params=param_info["trainable"],
        trainable_pct=param_info["trainable_pct"],
        train_loss=train_output.training_loss,
        eval_loss=eval_output["eval_loss"],
        train_time_seconds=elapsed,
        train_samples_per_second=train_output.metrics.get(
            "train_samples_per_second", 0
        ),
        log_history=list(trainer.state.log_history),
    )

    # Explicit cleanup to prevent OOM across sequential runs
    del trainer, train_output, eval_output
    import gc
    gc.collect()

    return result


def save_results(results: list[TrainResult], path: str = "results/ablation_results.csv"):
    """Save ablation results to CSV.

    Args:
        results: List of TrainResult from multiple runs.
        path: Output CSV path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "run_name", "lora_rank", "lora_alpha", "learning_rate",
        "num_epochs", "trainable_params", "trainable_pct",
        "train_loss", "eval_loss", "train_time_seconds",
        "train_samples_per_second",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


# --- Per-run persistence for crash-safe recovery ---

FIELDNAMES = [
    "run_name", "lora_rank", "lora_alpha", "learning_rate",
    "num_epochs", "trainable_params", "trainable_pct",
    "train_loss", "eval_loss", "train_time_seconds",
    "train_samples_per_second",
]


def save_run_result(
    result: TrainResult,
    runs_dir: str = "results/runs",
    csv_path: str = "results/ablation_results.csv",
) -> None:
    """Persist a single run immediately after training completes.

    Writes:
      - results/runs/{run_name}.json  (full metrics)
      - Appends one row to results/ablation_results.csv (creates with header if new)
    """
    os.makedirs(runs_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(runs_dir, f"{result.run_name}.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # CSV append
    csv_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(result.to_dict())


def load_completed_runs(runs_dir: str = "results/runs") -> dict[str, dict]:
    """Load all completed run results from per-run JSON files.

    Returns:
        Dict mapping run_name -> metrics dict.
    """
    completed = {}
    for path in sorted(glob(os.path.join(runs_dir, "*.json"))):
        fname = os.path.basename(path)
        if fname.endswith("_logs.json"):
            continue  # skip log history files
        with open(path) as f:
            data = json.load(f)
        completed[data["run_name"]] = data
    return completed


def save_log_history(
    run_name: str,
    log_history: list,
    runs_dir: str = "results/runs",
) -> None:
    """Save per-run training log history for curve reconstruction."""
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run_name}_logs.json")
    with open(path, "w") as f:
        json.dump(log_history, f, indent=2)
