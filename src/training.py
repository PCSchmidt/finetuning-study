"""Training loop for LoRA fine-tuning.

Uses HuggingFace Trainer with configurable hyperparameters.
Logs metrics to CSV for later visualization.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
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

    return TrainResult(
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
        log_history=trainer.state.log_history,
    )


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
