# Demo Guide — Parameter-Efficient Fine-Tuning Study (LoRA)

## Objective

This project investigates how LoRA rank and alpha choices affect fine-tuning quality on a summarization task. The central question is practical: given that rank-2 and rank-16 adapters both update far less than 2% of a model's parameters, does rank actually matter, and where does the benefit flatten out?

To answer that, five configurations spanning rank [2, 4, 8, 16] and alpha [8, 16, 32] were trained on GPT-2 (124M) for dialogue summarization using the SAMSum dataset. Every run was evaluated with ROUGE and BERTScore, and the results were visualized as ablation heatmaps and training curves. All five runs are CPU-only and fully reproducible from the notebook on a standard laptop.

## Experiment Design

The ablation varies two LoRA hyperparameters independently:

- **Rank** controls the dimensionality of the low-rank update matrices (A and B). Higher rank means more expressive capacity but more trainable parameters.
- **Alpha** is a scaling factor applied as alpha/rank. Holding rank constant and increasing alpha increases the effective learning rate of the adapter.

LoRA replaces full weight updates with the product of two small matrices. For a weight matrix W, the adapted output is:

```
output = W * x + (alpha / rank) * B * A * x
```

B and A are the only trained parameters. At rank 8 across all 24 attention projection layers in GPT-2, this yields 811,008 trainable parameters — 0.65% of the full model. At rank 2, that drops to 202,752. The original weights are frozen throughout.

Training used 500 SAMSum examples per run (chosen for CPU feasibility), 3 epochs, and a learning rate of 3e-4 with AdamW. Each run saves per-step loss history and a checkpoint for later evaluation.

## Navigating the Notebook

The notebook is organized in numbered sections that follow the experiment from setup through evaluation. The key sections are:

**Section 1 — Motivation** lays out the hypothesis and explains why rank is worth studying empirically rather than defaulting to a rule of thumb.

**Section 5 — LoRA Configuration** defines each of the five configurations and shows the parameter counts. This is where the ablation grid is established.

**Section 7 — Results Analysis** contains the heatmap, training loss curves, and the diminishing returns comparison. The heatmap plots rank x alpha against eval loss; the pattern of interest is that rank 8 and rank 16 differ by less than 0.001 in eval loss despite rank 16 using twice as many trainable parameters.

**Section 9 — Best Model Evaluation** applies the best checkpoint (rank 8, alpha 32) to the test set and compares it against base GPT-2 with no fine-tuning. ROUGE-L improves from 0.105 to 0.139; BERTScore F1 improves from 0.376 to 0.453. The qualitative comparison shows that base GPT-2 frequently generates off-topic or empty outputs, while the fine-tuned model consistently produces on-topic summaries.

The PDF version of the notebook includes all cell outputs, charts, and printed results, so you do not need to re-run anything to see the full results.

## Code Organization

The `src/` directory has five modules with clear separation of responsibilities:

- `data_prep.py` — loads and tokenizes SAMSum splits
- `lora_config.py` — defines LoRA configurations and builds adapted models via PEFT
- `training.py` — runs fine-tuning with HuggingFace Trainer; includes per-run JSON persistence for crash-safe recovery across sequential runs
- `evaluation.py` — computes ROUGE-1/2/L and BERTScore; generates the before/after comparison
- `visualization.py` — produces the ablation heatmaps, training curves, and parameter efficiency charts

12 unit tests cover data loading, configuration building, output shapes, and metric calculation. Tests run on every push via GitHub Actions CI.

## What the Results Mean

The main finding is that rank 8 is the practical ceiling for this task and dataset size. Rank 4 gets within 0.022 eval loss of rank 8 at half the parameter count. Moving to rank 16 recovers less than 0.001 additional loss improvement while adding 811K parameters. This pattern of diminishing returns is the expected theoretical behavior of low-rank approximations and the experiment confirms it empirically.

At 500 training examples, the absolute ROUGE scores are modest. With the full SAMSum dataset (14,000 examples) and GPU training, scores would improve substantially. The relative ordering of configurations — which ranks outperform others — would likely remain the same. The methodology itself scales directly to larger models: the same ablation design applied to Llama 3 or Mistral would follow identical steps, with GPU resources as the only change.

## Limitations and Honest Context

This is a coursework and learning project, not a production system. GPT-2 was chosen because it fits CPU training; it is not a competitive summarization model at this scale. The training set size of 500 examples was chosen for reproducibility on a laptop, which constrains absolute performance.

The study does not cover quantization, inference optimization, or serving — those are addressed separately in the Inference Optimization Study. The goal here is specifically to understand the relationship between rank choice and fine-tuning quality in a controlled setting.
