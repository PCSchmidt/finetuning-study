"""Visualization module for fine-tuning study.

Generates Plotly charts for training curves, ablation heatmaps,
and metric comparisons. Exports to PNG and interactive HTML.
"""

from __future__ import annotations

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_training_curves(
    log_histories: dict[str, list[dict]],
    output_path: str = "results/training_curves.png",
) -> go.Figure:
    """Plot training loss curves for multiple runs.

    Args:
        log_histories: Mapping of run_name -> trainer log_history list.
        output_path: Path to save PNG.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    for run_name, history in log_histories.items():
        steps = [h["step"] for h in history if "loss" in h]
        losses = [h["loss"] for h in history if "loss" in h]
        fig.add_trace(go.Scatter(
            x=steps, y=losses, mode="lines", name=run_name,
        ))

    fig.update_layout(
        title="Training Loss by LoRA Configuration",
        xaxis_title="Step",
        yaxis_title="Training Loss",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    _save_fig(fig, output_path)
    return fig


def plot_ablation_heatmap(
    results_df: pd.DataFrame,
    metric: str = "eval_loss",
    output_path: str = "results/ablation_heatmap.png",
) -> go.Figure:
    """Plot a heatmap of rank × alpha vs a metric.

    Args:
        results_df: DataFrame with lora_rank, lora_alpha, and metric columns.
        metric: Column name to visualize.
        output_path: Path to save PNG.

    Returns:
        Plotly Figure.
    """
    pivot = results_df.pivot_table(
        index="lora_rank", columns="lora_alpha", values=metric, aggfunc="mean",
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="Viridis",
        text=pivot.values.round(4),
        texttemplate="%{text}",
        colorbar_title=metric,
    ))

    fig.update_layout(
        title=f"LoRA Ablation: {metric}",
        xaxis_title="LoRA Alpha",
        yaxis_title="LoRA Rank",
        template="plotly_white",
    )

    _save_fig(fig, output_path)
    return fig


def plot_metric_comparison(
    base_metrics: dict,
    finetuned_metrics: dict,
    output_path: str = "results/metric_comparison.png",
) -> go.Figure:
    """Bar chart comparing base vs fine-tuned model metrics.

    Args:
        base_metrics: Metrics dict from base model evaluation.
        finetuned_metrics: Metrics dict from fine-tuned model evaluation.
        output_path: Path to save PNG.

    Returns:
        Plotly Figure.
    """
    # Select numeric metrics only
    metric_names = [k for k in base_metrics if k != "predictions" and isinstance(base_metrics[k], (int, float))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Base GPT-2",
        x=metric_names,
        y=[base_metrics[m] for m in metric_names],
    ))
    fig.add_trace(go.Bar(
        name="Fine-tuned (LoRA)",
        x=metric_names,
        y=[finetuned_metrics[m] for m in metric_names],
    ))

    fig.update_layout(
        title="Base vs Fine-Tuned Model Metrics",
        barmode="group",
        template="plotly_white",
        yaxis_title="Score",
    )

    _save_fig(fig, output_path)
    return fig


def plot_parameter_efficiency(
    results_df: pd.DataFrame,
    output_path: str = "results/parameter_efficiency.png",
) -> go.Figure:
    """Scatter plot of trainable parameters vs eval loss.

    Args:
        results_df: DataFrame with trainable_params and eval_loss columns.
        output_path: Path to save PNG.

    Returns:
        Plotly Figure.
    """
    fig = px.scatter(
        results_df,
        x="trainable_params",
        y="eval_loss",
        color="lora_rank",
        size="lora_alpha",
        hover_data=["run_name", "learning_rate"],
        title="Parameter Efficiency: Trainable Params vs Eval Loss",
        template="plotly_white",
        labels={
            "trainable_params": "Trainable Parameters",
            "eval_loss": "Eval Loss",
            "lora_rank": "LoRA Rank",
        },
    )

    _save_fig(fig, output_path)
    return fig


def _save_fig(fig: go.Figure, path: str):
    """Save figure to PNG (and HTML alongside).

    Args:
        fig: Plotly figure.
        path: Output PNG path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_image(path, width=1000, height=600, scale=2)
    html_path = path.replace(".png", ".html")
    fig.write_html(html_path)
