"""Pytest fixtures and mocks for Autotrain tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Configure pytest to mock unsloth and trl before any imports."""
    from types import ModuleType

    # Create mock unsloth module before any imports
    mock_model_instance = MagicMock()
    mock_model_instance.model = MagicMock()
    mock_model_instance.device = "cuda"
    mock_model_instance.generate = MagicMock(return_value=[[1, 2, 3]])
    mock_model_instance.save_pretrained = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(
        input_ids=MagicMock(to=MagicMock(return_value=MagicMock())), attention_mask=MagicMock()
    )
    mock_tokenizer.decode = MagicMock(return_value="Generated text")

    # Create FastLanguageModel class mock
    mock_fast_language_model = MagicMock()
    mock_fast_language_model.from_pretrained = MagicMock(
        return_value=(mock_model_instance, mock_tokenizer)
    )
    mock_fast_language_model.save_pretrained_gguf = MagicMock()
    mock_fast_language_model.get_peft_model = MagicMock(return_value=mock_model_instance)

    # Create FastVisionModel class mock
    mock_fast_vision_model = MagicMock()
    mock_fast_vision_model.from_pretrained = MagicMock(
        return_value=(mock_model_instance, mock_tokenizer)
    )
    mock_fast_vision_model.get_peft_model = MagicMock(return_value=mock_model_instance)
    mock_fast_vision_model.for_inference = MagicMock()

    # Create unsloth module mock as a proper ModuleType
    mock_unsloth_module = ModuleType("unsloth")
    mock_unsloth_module.FastLanguageModel = mock_fast_language_model
    mock_unsloth_module.FastVisionModel = mock_fast_vision_model
    mock_unsloth_module.__spec__ = MagicMock()

    # Insert mock into sys.modules before any imports
    sys.modules["unsloth"] = mock_unsloth_module

    # Create mock trl module
    mock_dpo_trainer_class = MagicMock()
    mock_dpo_trainer_instance = MagicMock()
    mock_dpo_trainer_instance.train = MagicMock()
    mock_dpo_trainer_class.return_value = mock_dpo_trainer_instance

    mock_sft_trainer_class = MagicMock()
    mock_sft_trainer_instance = MagicMock()
    mock_sft_trainer_instance.train = MagicMock()
    mock_sft_trainer_class.return_value = mock_sft_trainer_instance

    mock_trl_module = ModuleType("trl")
    # DISABLED: mock_trl_module.DPOTrainer = mock_dpo_trainer_class
    mock_trl_module.SFTTrainer = mock_sft_trainer_class
    mock_trl_module.__spec__ = MagicMock()

    sys.modules["trl"] = mock_trl_module


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_litellm():
    """Mock litellm completion calls."""
    with patch("litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mocked response from LLM"
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_unsloth():
    """Mock Unsloth FastLanguageModel."""
    # Return the already-mocked module from sys.modules
    yield sys.modules["unsloth"]


@pytest.fixture
def mock_trainer():
    """Mock TRL SFTTrainer."""
    with patch("trl.SFTTrainer") as mock_sft_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train = MagicMock()
        mock_sft_trainer.return_value = mock_trainer_instance
        yield mock_sft_trainer


# DISABLED: DPO-related fixture
# @pytest.fixture
# def mock_dpo_trainer():
#     """Mock TRL DPOTrainer."""
#     with patch("trl.DPOTrainer") as mock_dpo_trainer:
#         mock_trainer_instance = MagicMock()
#         mock_trainer_instance.train = MagicMock()
#         mock_dpo_trainer.return_value = mock_trainer_instance
#         yield mock_dpo_trainer


@pytest.fixture
def sample_data():
    """Sample training data for tests."""
    return [
        {"input": "What is 2+2?", "output": "2+2=4"},
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "Explain gravity", "output": "Gravity is a fundamental force of nature."},
    ]


@pytest.fixture
def benchmark_samples():
    """Sample benchmark data for tests."""
    return [
        {"input_data": "What is 10 × 10?", "expected_output": "100"},
        {"input_data": "What is H2O?", "expected_output": "Water"},
    ]


@pytest.fixture
def expert_config():
    """Default expert configuration for tests."""
    return {
        "model_name": "unsloth/Qwen3.5-27B-GGUF",
        "api_key": "test-api-key",
        "production_rate": 0.5,
    }


@pytest.fixture
def model_config():
    """Default model configuration for tests."""
    return {
        "model_name": "unsloth/Qwen3.5-27B-GGUF",
        "sample_multiplier": 2,
    }


@pytest.fixture
def cleanup_env():
    """Clean up environment variables after test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
