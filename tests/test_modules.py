"""Tests for finetuning-study modules.

Tests model loading, LoRA configuration, data preparation,
and parameter counting without requiring GPU or large downloads.
"""

import pytest


class TestLoraConfig:
    """Tests for LoRA configuration and model setup."""

    def test_create_lora_config_defaults(self):
        from src.lora_config import create_lora_config
        config = create_lora_config()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert "c_attn" in config.target_modules
        assert "c_proj" in config.target_modules

    def test_create_lora_config_custom(self):
        from src.lora_config import create_lora_config
        config = create_lora_config(rank=4, alpha=32, dropout=0.1)
        assert config.r == 4
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1

    def test_lora_parameter_count(self):
        from src.lora_config import lora_parameter_count
        # GPT-2: hidden_dim=768, 2 modules per layer, 12 layers = 24 modules
        count = lora_parameter_count(hidden_dim=768, rank=8, num_modules=24)
        # 24 * 2 * 768 * 8 = 294,912
        assert count == 294_912

    def test_lora_parameter_count_rank_2(self):
        from src.lora_config import lora_parameter_count
        count = lora_parameter_count(hidden_dim=768, rank=2, num_modules=24)
        assert count == 24 * 2 * 768 * 2

    def test_load_base_model(self):
        from src.lora_config import load_base_model
        model, tokenizer = load_base_model("gpt2")
        assert tokenizer.pad_token == tokenizer.eos_token
        assert model.config.pad_token_id == tokenizer.pad_token_id

    def test_apply_lora(self):
        from src.lora_config import load_base_model, create_lora_config, apply_lora, count_parameters
        model, _ = load_base_model("gpt2")
        config = create_lora_config(rank=2, alpha=8)
        peft_model = apply_lora(model, config)
        params = count_parameters(peft_model)
        assert params["trainable"] > 0
        assert params["trainable_pct"] < 1.0  # Should be well under 1%
        assert params["total"] > params["trainable"]


class TestDataPrep:
    """Tests for data preparation."""

    def test_format_prompt(self):
        from src.data_prep import format_prompt
        prompt = format_prompt("Alice: Hi\nBob: Hello")
        assert "Summarize the following dialogue" in prompt
        assert "Alice: Hi" in prompt
        assert prompt.endswith("Summary:")

    def test_format_prompt_empty(self):
        from src.data_prep import format_prompt
        prompt = format_prompt("")
        assert "Summary:" in prompt

    def test_prepare_dataset(self):
        from datasets import Dataset
        from src.data_prep import prepare_dataset
        from src.lora_config import load_base_model

        _, tokenizer = load_base_model("gpt2")
        ds = Dataset.from_dict({
            "dialogue": ["Alice: Hi\nBob: Hello"],
            "summary": ["Alice greets Bob."],
        })
        tokenized = prepare_dataset(ds, tokenizer, max_length=64)
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names
        assert "labels" in tokenized.column_names
        assert len(tokenized[0]["input_ids"]) == 64
        # Labels should have -100 for prompt tokens
        assert -100 in tokenized[0]["labels"]


class TestTraining:
    """Tests for training configuration."""

    def test_train_config_defaults(self):
        from src.training import TrainConfig
        cfg = TrainConfig()
        assert cfg.lora_rank == 8
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 4

    def test_train_result_fields(self):
        from src.training import TrainResult
        result = TrainResult(
            run_name="test",
            lora_rank=8,
            lora_alpha=16,
            learning_rate=3e-4,
            num_epochs=3,
            trainable_params=24576,
            trainable_pct=0.02,
            train_loss=2.5,
            eval_loss=2.8,
            train_time_seconds=60.0,
            train_samples_per_second=8.3,
        )
        assert result.run_name == "test"
        assert result.eval_loss == 2.8

    def test_save_results(self, tmp_path):
        from src.training import TrainResult, save_results
        results = [
            TrainResult(
                run_name="r8_a16", lora_rank=8, lora_alpha=16,
                learning_rate=3e-4, num_epochs=3, trainable_params=24576,
                trainable_pct=0.02, train_loss=2.5, eval_loss=2.8,
                train_time_seconds=60.0, train_samples_per_second=8.3,
            ),
        ]
        path = str(tmp_path / "test_results.csv")
        save_results(results, path)

        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["run_name"] == "r8_a16"
        assert df.iloc[0]["lora_rank"] == 8
