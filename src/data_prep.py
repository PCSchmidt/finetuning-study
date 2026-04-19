"""Data preparation for fine-tuning study.

Loads and preprocesses a summarization dataset into prompt/completion pairs
suitable for causal LM fine-tuning.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset


def load_samsum(split: str = "train", max_samples: int | None = None) -> Dataset:
    """Load SAMSum dialogue summarization dataset.

    Args:
        split: Dataset split ('train', 'validation', 'test').
        max_samples: Cap the number of examples. None = use all.

    Returns:
        HuggingFace Dataset with 'dialogue' and 'summary' columns.
    """
    ds = load_dataset("samsum", split=split, trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def format_prompt(dialogue: str) -> str:
    """Format a dialogue into an instruction prompt for causal LM."""
    return (
        "Summarize the following dialogue:\n\n"
        f"{dialogue}\n\n"
        "Summary:"
    )


def prepare_dataset(
    ds: Dataset,
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Tokenize a SAMSum dataset for causal LM training.

    Each example becomes: prompt + " " + summary + eos_token.
    Labels mask the prompt tokens with -100 so loss is only on the summary.

    Args:
        ds: Dataset with 'dialogue' and 'summary' columns.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset with input_ids, attention_mask, labels.
    """
    def tokenize(example):
        prompt = format_prompt(example["dialogue"])
        full_text = prompt + " " + example["summary"] + tokenizer.eos_token

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Tokenize prompt alone to find where summary starts
        prompt_tokenized = tokenizer(
            prompt + " ",
            truncation=True,
            max_length=max_length,
        )
        prompt_len = len(prompt_tokenized["input_ids"])

        # Mask prompt tokens in labels
        labels = tokenized["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        # Also mask padding
        labels = [
            -100 if tokenized["attention_mask"][i] == 0 else labels[i]
            for i in range(len(labels))
        ]
        tokenized["labels"] = labels
        return tokenized

    return ds.map(tokenize, remove_columns=ds.column_names)
