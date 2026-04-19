# Parameter-Efficient Fine-Tuning Study

LoRA rank ablation on GPT-2 for dialogue summarization — CPU-only, fully reproducible.

## What This Demonstrates

- **LoRA (Low-Rank Adaptation)** applied to GPT-2 (124M) for dialogue summarization
- **Systematic ablation** over rank [2, 4, 8, 16] and alpha [8, 16, 32]
- **ROUGE + BERTScore** evaluation with before/after comparison against zero-shot baseline
- **CPU-feasible training** — no GPU required, runs on commodity hardware
- **Clean experiment design**: isolated variables, reproducible results, professional visualization

## Key Results

| Config | Trainable Params | % of 124M | Eval Loss | ROUGE-L |
|--------|-----------------|-----------|-----------|---------|
| r2_a8  | ~6K             | 0.005%    | TBD       | TBD     |
| r4_a16 | ~12K            | 0.010%    | TBD       | TBD     |
| r8_a16 | ~25K            | 0.020%    | TBD       | TBD     |
| r8_a32 | ~25K            | 0.020%    | TBD       | TBD     |
| r16_a32| ~49K            | 0.040%    | TBD       | TBD     |

*Results populated after notebook execution.*

## Project Structure

```
finetuning-study/
├── src/
│   ├── data_prep.py        # Dataset loading, tokenization, prompt formatting
│   ├── lora_config.py      # LoRA configuration, model setup, parameter counting
│   ├── training.py         # Training loop with HF Trainer, CSV logging
│   ├── evaluation.py       # ROUGE, BERTScore, before/after comparisons
│   └── visualization.py    # Plotly charts: training curves, heatmaps
├── notebooks/
│   └── finetuning_study.ipynb  # Main study notebook (run this)
├── tests/
│   └── test_modules.py     # Unit tests for all modules
├── results/                # Generated: CSVs, PNGs, HTML charts
├── evidence/               # Generated: PDF export for portfolio
└── .github/workflows/ci.yml
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run the study notebook
jupyter notebook notebooks/finetuning_study.ipynb
```

## Technical Details

### LoRA Theory

Standard fine-tuning updates W ∈ ℝ^{d×d} directly. LoRA learns a low-rank update:

W' = W + (α/r) × B × A

where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, and r ≪ d. This constrains weight updates to a rank-r subspace, reducing trainable parameters by 95%+.

### Dataset

**SAMSum** — 16K dialogue/summary pairs. We use 500 train / 100 val / 100 test for CPU feasibility.

### Why GPT-2?

At 124M parameters, GPT-2 is the largest model that fine-tunes in reasonable time on CPU. The LoRA methodology is model-size-agnostic — the same ablation approach applies to Llama, Mistral, etc.

## Related Projects

- [Journal Summarizer](https://github.com/PCSchmidt/generative-ai-journal-summarizer) — Multi-provider LLM gateway with RAG pipeline
- [Inference Optimization Study](https://github.com/PCSchmidt/inference-optimization-study) — ONNX + quantization + batching benchmarks

## License

MIT
