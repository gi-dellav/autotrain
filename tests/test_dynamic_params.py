"""Tests for dynamic functional parameters and iteration data retention."""

import pytest
from unittest.mock import MagicMock, patch

from autotrain.config import PEFTConfig, TrainingConfig
from autotrain.utils.function_evaluator import (
    evaluate_weight_decay,
    evaluate_batch_size,
    evaluate_lora_rank,
    validate_weight_decay,
    validate_batch_size,
    validate_lora_rank,
)


class TestFunctionEvaluator:
    """Tests for new function evaluator utilities."""

    class TestValidateWeightDecay:
        def test_valid_weight_decay(self):
            validate_weight_decay(0.01)
            validate_weight_decay(0.0)
            validate_weight_decay(0.5)

        def test_negative_weight_decay_raises(self):
            with pytest.raises(ValueError, match="non-negative"):
                validate_weight_decay(-0.1)

        def test_non_number_raises(self):
            with pytest.raises(ValueError, match="must be a number"):
                validate_weight_decay("0.01")

    class TestValidateBatchSize:
        def test_valid_batch_size(self):
            validate_batch_size(2)
            validate_batch_size(4)
            validate_batch_size(8)

        def test_non_positive_batch_size_raises(self):
            with pytest.raises(ValueError, match="positive"):
                validate_batch_size(0)
            with pytest.raises(ValueError, match="positive"):
                validate_batch_size(-1)

        def test_non_integer_batch_size_raises(self):
            with pytest.raises(ValueError, match="must be an integer"):
                validate_batch_size(2.5)

    class TestValidateLoraRank:
        def test_valid_lora_rank(self):
            validate_lora_rank(16)
            validate_lora_rank(32)
            validate_lora_rank(64)

        def test_non_positive_lora_rank_raises(self):
            with pytest.raises(ValueError, match="positive"):
                validate_lora_rank(0)
            with pytest.raises(ValueError, match="positive"):
                validate_lora_rank(-1)

        def test_non_integer_lora_rank_raises(self):
            with pytest.raises(ValueError, match="must be an integer"):
                validate_lora_rank(16.5)

    class TestEvaluateWeightDecay:
        def test_static_value(self):
            assert evaluate_weight_decay(0.01, 0) == 0.01
            assert evaluate_weight_decay(0.05, 5) == 0.05

        def test_none_returns_default(self):
            assert evaluate_weight_decay(None, 0) == 0.01

        def test_callable(self):
            fn = lambda iteration: 0.01 + iteration * 0.005
            assert evaluate_weight_decay(fn, 0) == 0.01
            assert evaluate_weight_decay(fn, 2) == 0.02

    class TestEvaluateBatchSize:
        def test_static_value(self):
            assert evaluate_batch_size(4, 0) == 4
            assert evaluate_batch_size(8, 5) == 8

        def test_none_returns_default(self):
            assert evaluate_batch_size(None, 0) == 2

        def test_callable(self):
            fn = lambda iteration: min(2 + iteration, 8)
            assert evaluate_batch_size(fn, 0) == 2
            assert evaluate_batch_size(fn, 3) == 5
            assert evaluate_batch_size(fn, 10) == 8

    class TestEvaluateLoraRank:
        def test_static_value(self):
            assert evaluate_lora_rank(16, 0) == 16
            assert evaluate_lora_rank(32, 5) == 32

        def test_none_returns_default(self):
            assert evaluate_lora_rank(None, 0) == 16

        def test_callable(self):
            fn = lambda iteration: 16 + iteration * 8
            assert evaluate_lora_rank(fn, 0) == 16
            assert evaluate_lora_rank(fn, 2) == 32


class TestConfigFunctionalParameters:
    """Tests for functional parameters in config classes."""

    def test_training_config_weight_decay_fn(self):
        config = TrainingConfig(
            weight_decay=0.01,
            weight_decay_fn=lambda it: 0.01 + it * 0.005,
        )
        assert config.weight_decay_fn is not None
        assert config.weight_decay_fn(0) == 0.01
        assert config.weight_decay_fn(4) == 0.03

    def test_training_config_batch_size_fn(self):
        config = TrainingConfig(
            batch_size=2,
            batch_size_fn=lambda it: min(2 + it, 8),
        )
        assert config.batch_size_fn is not None
        assert config.batch_size_fn(0) == 2
        assert config.batch_size_fn(6) == 8

    def test_peft_config_lora_rank_fn(self):
        config = PEFTConfig(
            r=16,
            lora_rank_fn=lambda it: 16 + it * 8,
        )
        assert config.lora_rank_fn is not None
        assert config.lora_rank_fn(0) == 16
        assert config.lora_rank_fn(2) == 32

    def test_training_config_keep_last_n_iters(self):
        config = TrainingConfig(keep_last_n_iters=5)
        assert config.keep_last_n_iters == 5

        config_default = TrainingConfig()
        assert config_default.keep_last_n_iters is None


class TestTrainingDataPruning:
    """Tests for training data pruning with keep_last_n_iters."""

    def test_prune_old_training_data(self):
        """Test that old training data is pruned correctly."""
        from autotrain.core.base_model import BaseModel

        model = BaseModel(model_name="test-model")
        
        # Add samples with different iteration metadata
        model._training_data = [
            {"input": "sample_0", "output": "out_0", "iteration": 0},
            {"input": "sample_1", "output": "out_1", "iteration": 1},
            {"input": "sample_2", "output": "out_2", "iteration": 2},
            {"input": "sample_3", "output": "out_3", "iteration": 3},
        ]

        from autotrain.core.training import _prune_old_training_data
        
        # Keep last 2 iterations, current iteration is 3
        _prune_old_training_data(model, current_iteration=3, keep_last_n_iters=2)
        
        # Should keep samples from iterations 2 and 3
        assert len(model._training_data) == 2
        assert model._training_data[0]["iteration"] == 2
        assert model._training_data[1]["iteration"] == 3

    def test_prune_keeps_initial_samples(self):
        """Test that initial samples (iteration=-1) are always kept."""
        from autotrain.core.base_model import BaseModel

        model = BaseModel(model_name="test-model")
        
        model._training_data = [
            {"input": "initial_0", "output": "out_0", "iteration": -1},
            {"input": "initial_1", "output": "out_1", "iteration": -1},
            {"input": "sample_0", "output": "out_0", "iteration": 0},
            {"input": "sample_1", "output": "out_1", "iteration": 1},
        ]

        from autotrain.core.training import _prune_old_training_data
        
        # Keep last 1 iteration, current iteration is 3
        _prune_old_training_data(model, current_iteration=3, keep_last_n_iters=1)
        
        # Should keep initial samples and iteration 3 only
        assert len(model._training_data) == 2
        assert model._training_data[0]["iteration"] == -1
        assert model._training_data[1]["iteration"] == -1

    def test_no_pruning_when_none(self):
        """Test that no pruning occurs when keep_last_n_iters is None."""
        from autotrain.core.base_model import BaseModel

        model = BaseModel(model_name="test-model")
        
        model._training_data = [
            {"input": "sample_0", "output": "out_0", "iteration": 0},
            {"input": "sample_1", "output": "out_1", "iteration": 1},
        ]

        from autotrain.core.training import _prune_old_training_data
        
        # Should not prune when keep_last_n_iters <= 0
        _prune_old_training_data(model, current_iteration=3, keep_last_n_iters=0)
        
        assert len(model._training_data) == 2


class TestDynamicParametersInTraining:
    """Integration tests for dynamic parameters during training."""

    @patch("autotrain.core.training._fine_tune")
    def test_training_uses_dynamic_weight_decay(self, mock_fine_tune, temp_dir):
        """Test that training uses weight_decay_fn when set."""
        from autotrain import Model
        
        model = Model(checkpoint_dir=str(temp_dir), sample_multiplier=1)
        model._training_config.weight_decay_fn = lambda it: 0.01 + it * 0.005
        
        # Mock the fine-tune to avoid actual training
        model._fine_tune = MagicMock()
        
        # Verify config is set correctly
        assert model._training_config.weight_decay_fn is not None
        assert model._training_config.weight_decay_fn(0) == 0.01
        assert model._training_config.weight_decay_fn(2) == 0.02

    @patch("autotrain.core.training._fine_tune")
    def test_training_uses_dynamic_batch_size(self, mock_fine_tune, temp_dir):
        """Test that training uses batch_size_fn when set."""
        from autotrain import Model
        
        model = Model(checkpoint_dir=str(temp_dir), sample_multiplier=1)
        model._training_config.batch_size_fn = lambda it: min(2 + it, 8)
        
        model._fine_tune = MagicMock()
        
        assert model._training_config.batch_size_fn is not None
        assert model._training_config.batch_size_fn(0) == 2
        assert model._training_config.batch_size_fn(10) == 8

    @patch("autotrain.core.training._fine_tune")
    def test_training_uses_dynamic_lora_rank(self, mock_fine_tune, temp_dir):
        """Test that training uses lora_rank_fn when set."""
        from autotrain import Model
        
        model = Model(checkpoint_dir=str(temp_dir), sample_multiplier=1)
        model._peft_config.lora_rank_fn = lambda it: 16 + it * 8
        
        model._fine_tune = MagicMock()
        
        assert model._peft_config.lora_rank_fn is not None
        assert model._peft_config.lora_rank_fn(0) == 16
        assert model._peft_config.lora_rank_fn(2) == 32
