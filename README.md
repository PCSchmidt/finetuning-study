# Parameter-Efficient Fine-Tuning Study

Systematic LoRA rank ablation on GPT-2 for dialogue summarization, run entirely on CPU.

## What This Demonstrates

- **LoRA (Low-Rank Adaptation)** applied to GPT-2 (124M parameters) for the SAMSum dialogue summarization task
- **Systematic ablation** over five configurations spanning rank [2, 4, 8, 16] and alpha [8, 16, 32]
- **ROUGE and BERTScore evaluation** with a before-and-after comparison against a zero-shot baseline
- **CPU-only training** showing that meaningful fine-tuning research is feasible on a standard laptop
- **Reproducible experiment design**: one variable changed at a time, version-controlled code, and documented methodology

## Key Results

The best configuration (rank 16, alpha 32) improved ROUGE-L from 0.105 to 0.139 and BERTScore F1 from 0.376 to 0.453 relative to zero-shot GPT-2. Most of this gain was already present at rank 8: moving from rank 8 to rank 16 adds 811K more trainable parameters while improving eval loss by less than 0.001 points.

### Ablation Summary

| Config  | Rank | Alpha | Trainable Params | % of 124M | Eval Loss |
|---------|------|-------|-----------------|-----------|-----------|
| r2_a8   | 2    | 8     | 202,752         | 0.16%     | 2.561     |
| r4_a16  | 4    | 16    | 405,504         | 0.33%     | 2.533     |
| r8_a16  | 8    | 16    | 811,008         | 0.65%     | 2.533     |
| r8_a32  | 8    | 32    | 811,008         | 0.65%     | 2.511     |
| r16_a32 | 16   | 32    | 1,622,016       | 1.29%     | 2.510     |

### Best Model vs. Zero-Shot Baseline

| Metric            | Base GPT-2 | Fine-tuned | Change  |
|-------------------|------------|------------|---------|
| ROUGE-1           | 0.119      | 0.166      | +0.047  |
| ROUGE-2           | 0.014      | 0.033      | +0.019  |
| ROUGE-L           | 0.105      | 0.139      | +0.034  |
| BERTScore F1      | 0.376      | 0.453      | +0.076  |

Evaluated on 100 SAMSum test examples. The base model occasionally produces empty outputs (counted as zero in ROUGE). The fine-tuned model generates a coherent summary for every example.

## Project Structure

```
finetuning-study/
├── src/
│   ├── data_prep.py        # Dataset loading, tokenization, prompt formatting
│   ├── lora_config.py      # LoRA configuration, model setup, parameter counting
│   ├── training.py         # Training loop with HuggingFace Trainer, CSV logging
│   ├── evaluation.py       # ROUGE, BERTScore, before/after comparisons
│   └── visualization.py    # Plotly charts: training curves, heatmaps
├── notebooks/
│   └── finetuning_study.ipynb  # Main study notebook (run this)
│   └── finetuning_study.pdf    # Executed notebook export
├── tests/
│   └── test_modules.py     # Unit tests for all modules
├── results/                # Generated: CSVs, PNGs, HTML charts
└── .github/workflows/ci.yml
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Open the study notebook
jupyter notebook notebooks/finetuning_study.ipynb
```

## Technical Details

### LoRA Theory

Standard fine-tuning updates every weight in the model, which requires storing gradients and optimizer state for hundreds of millions of parameters. LoRA freezes the original pretrained weights entirely and instead injects small trainable matrices into selected layers. For a weight matrix W, the adapted forward pass computes:

```
output = W * input + (alpha / rank) * B * A * input
```

where A and B are low-rank matrices. A has shape (rank x d) and B has shape (d x rank), so the number of new parameters per layer is 2 * d * rank rather than d * d. For GPT-2's hidden dimension of 768 and rank 8, that is roughly 12K parameters per targeted weight matrix rather than 590K. Applied across all attention layers in all 12 transformer blocks, the total trainable count is 811K out of 124M, or 0.65% of the model.

The scaling factor alpha controls how much the adapter output influences the model's predictions at initialization. The ratio alpha/rank is sometimes described as the effective learning rate of the adapter. Keeping alpha proportional to rank (for example, alpha=16 with rank=8, and alpha=32 with rank=16) is a common baseline, and this study includes configurations that vary them independently to see whether the absolute values or the ratio matters more.

### Dataset

SAMSum contains roughly 16,000 messenger-style conversation and summary pairs collected to study abstractive summarization. This study uses 500 training examples, 100 validation examples, and 100 test examples. That subset is large enough to produce measurable fine-tuning signal but small enough to keep each training run under two hours on a standard laptop CPU.

### Why GPT-2?

GPT-2 at 124M parameters is the largest transformer model that completes a three-epoch fine-tuning run in a reasonable amount of time on CPU. The LoRA methodology is not specific to GPT-2: the same ablation design applies directly to Llama, Mistral, or any other transformer when GPU resources are available. GPT-2 makes it possible to run the full study, inspect all intermediate results, and reproduce everything without cloud access.

## Related Projects

- [Journal Summarizer](https://github.com/PCSchmidt/generative-ai-journal-summarizer): Multi-provider LLM API gateway that originally motivated this study
- [Inference Optimization Study](https://github.com/PCSchmidt/inference-optimization-study): ONNX export, INT8 quantization, and adaptive batching benchmarks on a sentence transformer

## License

MIT
