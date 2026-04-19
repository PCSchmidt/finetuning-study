"""LoRA configuration and model setup.

Provides functions to load GPT-2 with LoRA adapters at various ranks,
with mathematical context connecting LoRA to low-rank matrix approximation.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_base_model(model_name: str = "gpt2"):
    """Load base model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Create a LoRA configuration.

    The key relationship: effective learning rate scales as alpha/rank.
    This is analogous to choosing the number of singular values to retain
    in a truncated SVD — rank controls capacity, alpha controls magnitude.

    Args:
        rank: LoRA rank (r). Number of dimensions in the low-rank decomposition.
              Analogous to keeping the top-r singular values in SVD.
        alpha: LoRA scaling factor. The adapter weight is scaled by alpha/rank.
        dropout: Dropout probability for LoRA layers.
        target_modules: Which linear layers to apply LoRA to.
              Defaults to attention projection matrices.

    Returns:
        LoraConfig instance.
    """
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )


def apply_lora(model, lora_config: LoraConfig):
    """Wrap a base model with LoRA adapters.

    Args:
        model: Base HuggingFace model.
        lora_config: LoRA configuration.

    Returns:
        PEFT model with LoRA adapters.
    """
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def count_parameters(model) -> dict:
    """Count trainable vs total parameters.

    Returns:
        Dict with 'total', 'trainable', 'trainable_pct' keys.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
    }


def lora_parameter_count(
    hidden_dim: int,
    rank: int,
    num_modules: int,
) -> int:
    """Calculate theoretical LoRA parameter count.

    Each adapted module adds two matrices: A (hidden_dim × rank) and B (rank × hidden_dim).
    Total added parameters = num_modules × 2 × hidden_dim × rank.

    This is the same as saying: we approximate the weight update ΔW ≈ BA,
    where B ∈ R^{d×r} and A ∈ R^{r×d}, constraining the update to a
    rank-r subspace of R^{d×d}.

    Args:
        hidden_dim: Model hidden dimension (768 for GPT-2).
        rank: LoRA rank.
        num_modules: Number of modules with LoRA adapters.

    Returns:
        Total number of added parameters.
    """
    return num_modules * 2 * hidden_dim * rank
