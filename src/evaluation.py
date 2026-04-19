"""Evaluation module for fine-tuned models.

Computes ROUGE, BERTScore, and generates before/after comparison tables.
"""

from __future__ import annotations

import torch
import pandas as pd
from tqdm import tqdm


def generate_summaries(
    model,
    tokenizer,
    dialogues: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """Generate summaries for a list of dialogues.

    Args:
        model: Fine-tuned (or base) model.
        tokenizer: Tokenizer.
        dialogues: List of dialogue strings.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of generated summary strings.
    """
    from src.data_prep import format_prompt

    model.eval()
    summaries = []

    for dialogue in tqdm(dialogues, desc="Generating summaries"):
        prompt = format_prompt(dialogue)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract everything after "Summary:"
        if "Summary:" in full_text:
            summary = full_text.split("Summary:")[-1].strip()
        else:
            summary = full_text[len(prompt):].strip()

        summaries.append(summary)

    return summaries


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores.

    Args:
        predictions: Generated summaries.
        references: Ground truth summaries.

    Returns:
        Dict with rouge1, rouge2, rougeL (each with precision, recall, fmeasure).
    """
    import evaluate

    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True,
    )
    return results


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict:
    """Compute BERTScore for semantic similarity.

    Args:
        predictions: Generated summaries.
        references: Ground truth summaries.
        model_type: BERTScore model. Uses DeBERTa for best accuracy.

    Returns:
        Dict with precision, recall, f1 (averaged).
    """
    import evaluate

    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=model_type,
    )
    return {
        "bertscore_precision": sum(results["precision"]) / len(results["precision"]),
        "bertscore_recall": sum(results["recall"]) / len(results["recall"]),
        "bertscore_f1": sum(results["f1"]) / len(results["f1"]),
    }


def evaluate_model(
    model,
    tokenizer,
    test_dialogues: list[str],
    test_references: list[str],
    max_new_tokens: int = 128,
    compute_bert: bool = True,
) -> dict:
    """Full evaluation pipeline: generate + score.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        test_dialogues: Dialogue inputs.
        test_references: Ground truth summaries.
        max_new_tokens: Max generation length.
        compute_bert: Whether to compute BERTScore (slower).

    Returns:
        Dict with all metrics and generated summaries.
    """
    predictions = generate_summaries(model, tokenizer, test_dialogues, max_new_tokens)

    metrics = compute_rouge(predictions, test_references)

    if compute_bert:
        bert_metrics = compute_bertscore(predictions, test_references)
        metrics.update(bert_metrics)

    metrics["predictions"] = predictions
    return metrics


def comparison_table(
    base_metrics: dict,
    finetuned_metrics: dict,
    dialogues: list[str],
    references: list[str],
    n_examples: int = 5,
) -> pd.DataFrame:
    """Create a before/after comparison DataFrame.

    Args:
        base_metrics: Metrics dict from evaluate_model on base model.
        finetuned_metrics: Metrics dict from evaluate_model on fine-tuned model.
        dialogues: Input dialogues.
        references: Ground truth summaries.
        n_examples: Number of examples to show.

    Returns:
        DataFrame with dialogue, reference, base_summary, finetuned_summary columns.
    """
    rows = []
    for i in range(min(n_examples, len(dialogues))):
        rows.append({
            "dialogue": dialogues[i][:200] + "..." if len(dialogues[i]) > 200 else dialogues[i],
            "reference": references[i],
            "base_summary": base_metrics["predictions"][i],
            "finetuned_summary": finetuned_metrics["predictions"][i],
        })
    return pd.DataFrame(rows)
