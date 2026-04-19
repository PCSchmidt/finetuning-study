# Demo Guide — Fine-Tuning Study

## 30-Second Pitch

> "I ran a systematic LoRA rank ablation on GPT-2 for dialogue summarization — entirely on CPU.
> The study shows that even rank-2 adapters with ~6K trainable parameters substantially beat
> zero-shot, and rank-8 captures most of the fine-tuning benefit at 0.02% of model parameters.
> All results are reproducible from the notebook."

## What to Show

### 1. The Ablation Design (30s)
Open the notebook and scroll to Section 1 (Motivation). Point out:
- The LoRA math: W' = W + (α/r)BA
- The ablation table: 5 configurations varying rank and alpha
- The theoretical parameter counts

### 2. Training Results (30s)
Scroll to Section 7 (Results Analysis). Highlight:
- Training loss curves showing convergence across configurations
- The heatmap showing rank × alpha vs eval loss
- Diminishing returns beyond rank 8

### 3. Before/After Comparison (30s)
Scroll to Section 9 (Best Model Evaluation). Show:
- The metric comparison bar chart (base vs fine-tuned)
- Side-by-side example outputs showing quality improvement
- ROUGE and BERTScore improvements

### 4. Code Quality (15s)
Show the `src/` directory structure. Highlight:
- Clean separation: data_prep, lora_config, training, evaluation, visualization
- Tests passing in CI
- Reproducibility: pyproject.toml with pinned dependencies

## Key Talking Points

1. **"The principles scale, not just the model."** The same rank ablation methodology works on
   Llama 7B, Mistral, or any transformer. GPT-2 demonstrates the methodology on CPU.

2. **"I understand the math."** α/r controls effective adaptation strength. Configurations with
   the same α/r ratio behave similarly regardless of absolute values.

3. **"I design experiments, not just run code."** The study has a clear hypothesis, controlled
   variables, and quantitative evaluation. This is how you validate ML decisions in production.

4. **"CPU-feasible means reproducible."** Anyone can clone this repo and replicate the results
   without needing GPU access or cloud credits.

## FAQ

**Q: Why not use a larger model?**
A: CPU constraint. But the methodology is the same — rank ablation is model-size-agnostic.
Production teams would apply this to Llama/Mistral on GPU.

**Q: Would more data help?**
A: Yes. We use 500 examples for CPU feasibility. With full SAMSum (14K) and GPU, expect
significantly better absolute scores while the relative rank ordering likely holds.

**Q: How does this relate to Journal Summarizer?**
A: Journal Summarizer uses API-based LLMs. This study explores when a fine-tuned local model
could replace API calls — lower latency, lower cost, better privacy.
