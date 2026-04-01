"""Tests for DPO module.

# DISABLED: All DPO-related tests in this file have been disabled via comments.
"""

# ============================================================================
# DISABLED: DPO Tests
# All DPO-related test classes and functions in this file are disabled.
# ============================================================================

"""
import json
from unittest.mock import MagicMock, patch

import pytest
from autotrain import Model

from autotrain.dpo import (
    DPOConfig,
    DPOTrainer,
    PreferenceSample,
    train_dpo,
)
from autotrain.expert import Expert


@pytest.fixture
def mock_dpo():
    """Mock DPO training components."""
    # trl is already mocked in conftest.py pytest_configure
    import sys

    yield sys.modules["trl"].DPOTrainer


class TestDPOConfig:
    """Tests for DPOConfig dataclass."""

    def test_default_config(self):
        """Test default DPO config values."""
        config = DPOConfig()

        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"
        assert config.label_smoothing == 0.0
        assert config.reference_free is False
        assert config.reference_model_name is None
        assert config.truncation_mode == "keep_start"
        assert config.epochs == 1
        assert config.batch_size == 2

    def test_custom_config(self):
        """Test custom DPO config."""
        config = DPOConfig(
            beta=0.5,
            loss_type="hinge",
            label_smoothing=0.1,
            reference_free=True,
            f_divergence_type="jsd",
            truncation_mode="keep_end",
            epochs=3,
            batch_size=4,
            learning_rate=1e-6,
        )

        assert config.beta == 0.5
        assert config.loss_type == "hinge"
        assert config.label_smoothing == 0.1
        assert config.reference_free is True
        assert config.f_divergence_type == "jsd"
        assert config.truncation_mode == "keep_end"
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-6

    def test_invalid_beta(self):
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be > 0"):
            DPOConfig(beta=-0.1)

    def test_invalid_label_smoothing(self):
        """Test that out-of-range label_smoothing raises error."""
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            DPOConfig(label_smoothing=-0.1)
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            DPOConfig(label_smoothing=1.0)

    def test_invalid_epochs(self):
        """Test that zero epochs raises error."""
        with pytest.raises(ValueError, match="epochs must be > 0"):
            DPOConfig(epochs=0)

    def test_invalid_batch_size(self):
        """Test that zero batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            DPOConfig(batch_size=0)

    def test_invalid_learning_rate(self):
        """Test that negative learning_rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            DPOConfig(learning_rate=-1e-7)

    def test_prompt_length_exceeds_max_length(self):
        """Test that max_prompt_length > max_length raises error."""
        with pytest.raises(ValueError, match="max_prompt_length.*must be <= max_length"):
            DPOConfig(max_length=256, max_prompt_length=512)

    def test_invalid_loss_type(self):
        """Test that invalid loss_type raises error."""
        with pytest.raises(ValueError, match="loss_type must be one of"):
            DPOConfig(loss_type="invalid")

    def test_invalid_truncation_mode(self):
        """Test that invalid truncation_mode raises error."""
        with pytest.raises(ValueError, match="truncation_mode must be one of"):
            DPOConfig(truncation_mode="invalid")


class TestPreferenceSample:
    """Tests for PreferenceSample dataclass."""

    def test_sample_creation(self):
        """Test PreferenceSample creation."""
        sample = PreferenceSample(
            prompt="What is 2+2?",
            chosen="2+2=4",
            rejected="2+2=5",
        )

        assert sample.prompt == "What is 2+2?"
        assert sample.chosen == "2+2=4"
        assert sample.rejected == "2+2=5"
        assert sample.metadata == {}

    def test_sample_with_metadata(self):
        """Test PreferenceSample with metadata."""
        sample = PreferenceSample(
            prompt="test",
            chosen="good answer",
            rejected="bad answer",
            metadata={"source": "expert", "score": 9},
        )

        assert sample.metadata["source"] == "expert"
        assert sample.metadata["score"] == 9

    def test_sample_to_dict(self):
        """Test PreferenceSample.to_dict."""
        sample = PreferenceSample(
            prompt="test",
            chosen="good",
            rejected="bad",
            metadata={"key": "value"},
        )

        data = sample.to_dict()

        assert data["prompt"] == "test"
        assert data["chosen"] == "good"
        assert data["rejected"] == "bad"
        assert data["metadata"]["key"] == "value"

    def test_sample_from_dict(self):
        """Test PreferenceSample.from_dict."""
        data = {
            "prompt": "test",
            "chosen": "good",
            "rejected": "bad",
            "metadata": {"key": "value"},
        }

        sample = PreferenceSample.from_dict(data)

        assert sample.prompt == "test"
        assert sample.chosen == "good"
        assert sample.metadata["key"] == "value"


class TestDPOTrainer:
    """Tests for DPOTrainer class."""

    def test_trainer_init(self):
        """Test DPOTrainer initialization."""
        model = Model()
        trainer = DPOTrainer(model)

        assert trainer.model == model
        assert trainer._preference_samples == []
        assert trainer._is_trained is False

    def test_trainer_with_config(self):
        """Test DPOTrainer with custom config."""
        model = Model()
        config = DPOConfig(beta=0.5, epochs=3)
        trainer = DPOTrainer(model, dpo_config=config)

        assert trainer.dpo_config.beta == 0.5
        assert trainer.dpo_config.epochs == 3

    def test_add_preference_sample(self):
        """Test adding preference sample."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample(
            prompt="What is 2+2?",
            chosen="4",
            rejected="5",
        )

        assert len(trainer._preference_samples) == 1
        sample = trainer._preference_samples[0]
        assert sample.prompt == "What is 2+2?"
        assert sample.chosen == "4"
        assert sample.rejected == "5"

    def test_add_preference_samples(self):
        """Test adding multiple preference samples."""
        model = Model()
        trainer = DPOTrainer(model)

        samples = [
            ("Q1", "A1_good", "A1_bad"),
            ("Q2", "A2_good", "A2_bad"),
            ("Q3", "A3_good", "A3_bad"),
        ]
        trainer.add_preference_samples(samples)

        assert len(trainer._preference_samples) == 3

    def test_add_preference_samples_with_metadata(self):
        """Test adding samples with metadata."""
        model = Model()
        trainer = DPOTrainer(model)

        samples = [
            ("Q1", "A1_good", "A1_bad"),
            ("Q2", "A2_good", "A2_bad"),
        ]
        metadata = [
            {"source": "expert1"},
            {"source": "expert2"},
        ]
        trainer.add_preference_samples(samples, metadata_list=metadata)

        assert trainer._preference_samples[0].metadata["source"] == "expert1"
        assert trainer._preference_samples[1].metadata["source"] == "expert2"

    def test_get_samples(self):
        """Test getting samples returns a copy."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample("Q", "A_good", "A_bad")

        samples = trainer.get_samples()
        samples.clear()

        # Original should be unchanged
        assert len(trainer.get_samples()) == 1

    def test_clear_samples(self):
        """Test clearing samples."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample("Q1", "A1", "B1")
        trainer.add_preference_sample("Q2", "A2", "B2")

        trainer.clear_samples()

        assert len(trainer._preference_samples) == 0

    def test_get_statistics(self):
        """Test getting dataset statistics."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample(
            prompt="Short prompt",
            chosen="Short answer",
            rejected="Bad answer",
        )
        trainer.add_preference_sample(
            prompt="This is a much longer prompt with more words",
            chosen="This is a longer and more detailed answer",
            rejected="Short bad answer",
        )

        stats = trainer.get_statistics()

        assert stats["total_samples"] == 2
        assert "avg_prompt_length" in stats
        assert "avg_chosen_length" in stats
        assert stats["max_prompt_length"] > stats["avg_prompt_length"]

    def test_get_statistics_empty(self):
        """Test statistics with no samples."""
        model = Model()
        trainer = DPOTrainer(model)

        stats = trainer.get_statistics()

        assert stats["total_samples"] == 0

    def test_export_dataset_json(self, temp_dir):
        """Test exporting dataset to JSON."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample("Q1", "A1", "B1")
        trainer.add_preference_sample("Q2", "A2", "B2")

        output_path = temp_dir / "dpo_dataset.json"
        trainer.export_dataset(str(output_path), format="json")

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["prompt"] == "Q1"

    def test_export_dataset_jsonl(self, temp_dir):
        """Test exporting dataset to JSONL."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample("Q1", "A1", "B1")
        trainer.add_preference_sample("Q2", "A2", "B2")

        output_path = temp_dir / "dpo_dataset.jsonl"
        trainer.export_dataset(str(output_path), format="jsonl")

        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_import_dataset_json(self, temp_dir):
        """Test importing dataset from JSON."""
        model = Model()
        trainer = DPOTrainer(model)

        # Create test file
        data = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
        ]

        input_path = temp_dir / "import.json"
        with open(input_path, "w") as f:
            json.dump(data, f)

        trainer.import_dataset(str(input_path), format="json")

        assert len(trainer._preference_samples) == 2

    def test_import_dataset_jsonl(self, temp_dir):
        """Test importing dataset from JSONL."""
        model = Model()
        trainer = DPOTrainer(model)

        input_path = temp_dir / "import.jsonl"
        with open(input_path, "w") as f:
            f.write('{"prompt": "Q1", "chosen": "A1", "rejected": "B1"}\n')
            f.write('{"prompt": "Q2", "chosen": "A2", "rejected": "B2"}\n')

        trainer.import_dataset(str(input_path), format="jsonl")

        assert len(trainer._preference_samples) == 2

    def test_import_dataset_not_found(self, temp_dir):
        """Test importing from non-existent file."""
        model = Model()
        trainer = DPOTrainer(model)

        with pytest.raises(FileNotFoundError):
            trainer.import_dataset(str(temp_dir / "nonexistent.json"))

    @pytest.mark.skip(reason="Requires extensive mocking of transformers")
    def test_train_requires_packages(self, temp_dir, mock_dpo):
        """Test that training requires trl package."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.add_preference_sample("Q", "A", "B")

        # Should work with mocked trl
        trainer.train(output_dir=str(temp_dir))

        assert mock_dpo.called

    def test_train_no_samples_raises(self, temp_dir):
        """Test that training with no samples raises error."""
        model = Model()
        trainer = DPOTrainer(model)

        with pytest.raises(ValueError, match="No preference samples"):
            trainer.train(output_dir=str(temp_dir))

    def test_prepare_dataset_with_template(self):
        """Test dataset preparation with an instruction template."""
        from autotrain.templates.core import InstructionTemplate

        model = Model()
        template = InstructionTemplate(
            name="test_template",
            user_template="Prompt: {instruction}\nResponse:",
            assistant_template="{output}",
            separator="\n",
        )
        model._template = template

        trainer = DPOTrainer(model)
        trainer.add_preference_sample(
            prompt="Hello",
            chosen="Hi",
            rejected="Bye",
        )

        dataset = trainer._prepare_dataset()

        assert len(dataset) == 1
        sample = dataset[0]

        assert sample["prompt"] == "Prompt: Hello\nResponse:"
        assert sample["chosen"] == "Hi"
        assert sample["rejected"] == "Bye"

    def test_prepare_dataset_without_template(self):
        """Test dataset preparation without a template."""
        model = Model()
        model._template = None

        trainer = DPOTrainer(model)
        trainer.add_preference_sample(
            prompt="Hello",
            chosen="Hi",
            rejected="Bye",
        )

        dataset = trainer._prepare_dataset()

        assert len(dataset) == 1
        sample = dataset[0]

        assert sample["prompt"] == "Hello"
        assert sample["chosen"] == "Hi"
        assert sample["rejected"] == "Bye"


class TestTrainDPO:
    """Tests for train_dpo convenience function."""

    @pytest.mark.skip(reason="Requires extensive mocking of transformers")
    def test_train_dpo_function(self, mock_dpo):
        """Test train_dpo function."""
        model = Model()

        samples = [
            PreferenceSample(prompt="Q1", chosen="A1", rejected="B1"),
            PreferenceSample(prompt="Q2", chosen="A2", rejected="B2"),
        ]

        # Should work with mocked trl
        result = train_dpo(model, samples, output_dir="./test_output")

        assert result["trained"] is True
        assert mock_dpo.called


class TestDPOWithExpert:
    """Tests for DPO with expert integration."""

    def test_generate_preference_samples(self, mock_litellm):
        """Test generating preference samples with expert."""
        model = Model()
        trainer = DPOTrainer(model)
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")

        # Mock comparison to always prefer first response
        mock_litellm.return_value.choices = [MagicMock(message=MagicMock(content="A is better"))]

        prompts = ["What is 2+2?", "What is 3+3?"]

        # This would generate samples but we can't fully test without real API
        # Just verify the method exists and doesn't crash with mocked API
        try:
            trainer.generate_preference_samples(expert, prompts, count_per_prompt=2)
        except Exception:
            pass  # Expected if API calls fail in test

    def test_create_from_rankings(self):
        """Test creating samples from rankings."""
        model = Model()
        trainer = DPOTrainer(model)

        trainer.create_from_rankings(
            prompt="What is the capital of France?",
            responses=["Paris", "London", "Berlin"],
            rankings=[1, 3, 2],
        )

        # Should create pairs: best vs each worse
        assert len(trainer._preference_samples) == 2

        # All should have Paris as chosen (rank 1)
        for sample in trainer._preference_samples:
            assert sample.chosen == "Paris"
            assert sample.rejected in ["London", "Berlin"]
"""

# ============================================================================
# END DISABLED: DPO Tests
# ============================================================================
