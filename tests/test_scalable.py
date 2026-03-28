"""Tests for scalable training features."""

import pytest
from autotrain import Model, ScalableTrainingConfig


class TestScalableTrainingConfig:
    """Tests for ScalableTrainingConfig dataclass."""

    def test_default_config(self):
        """Test default scalable config values."""
        config = ScalableTrainingConfig()

        assert config.gradient_checkpointing == "unsloth"
        assert config.mixed_precision == "bf16"
        assert config.batch_size_auto_tune is False
        assert config.max_memory_mb is None
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.use_flash_attention is False

    def test_custom_config(self):
        """Test custom scalable config."""
        config = ScalableTrainingConfig(
            gradient_checkpointing=False,
            mixed_precision="bf16",
            batch_size_auto_tune=True,
            max_memory_mb=16384,
            num_workers=8,
            pin_memory=False,
        )

        assert config.gradient_checkpointing is False
        assert config.mixed_precision == "bf16"
        assert config.batch_size_auto_tune is True
        assert config.max_memory_mb == 16384
        assert config.num_workers == 8
        assert config.pin_memory is False

    def test_invalid_mixed_precision(self):
        """Test that invalid mixed_precision raises error."""
        with pytest.raises(ValueError, match="mixed_precision must be"):
            ScalableTrainingConfig(mixed_precision="invalid")

    def test_invalid_num_workers(self):
        """Test that negative num_workers raises error."""
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            ScalableTrainingConfig(num_workers=-1)

    def test_invalid_max_memory(self):
        """Test that invalid max_memory_mb raises error."""
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            ScalableTrainingConfig(max_memory_mb=-100)


class TestModelScalableConfig:
    """Tests for Model scalable training integration."""

    def test_model_default_scalable_config(self):
        """Test Model has default scalable config."""
        model = Model()

        assert model._scalable_config is not None
        assert model._scalable_config.gradient_checkpointing == "unsloth"

    def test_model_custom_scalable_config(self):
        """Test Model with custom scalable config."""
        config = ScalableTrainingConfig(
            gradient_checkpointing=False,
            mixed_precision="bf16",
            batch_size_auto_tune=True,
        )
        model = Model(scalable_config=config)

        assert model._scalable_config.gradient_checkpointing is False
        assert model._scalable_config.mixed_precision == "bf16"
        assert model._scalable_config.batch_size_auto_tune is True

    def test_set_scalable_config(self):
        """Test setting scalable config."""
        model = Model()
        model.set_scalable_config(
            gradient_checkpointing=False,
            mixed_precision="bf16",
            num_workers=8,
        )

        config = model.get_scalable_config()
        assert config.gradient_checkpointing is False
        assert config.mixed_precision == "bf16"
        assert config.num_workers == 8

    def test_get_scalable_config(self):
        """Test getting scalable config."""
        model = Model()
        config = model.get_scalable_config()

        assert isinstance(config, ScalableTrainingConfig)
        assert config.gradient_checkpointing == "unsloth"

    def test_auto_tune_batch_size_no_model_loaded(self):
        """Test auto-tune when model not loaded."""
        model = Model()

        # Should return default batch size when model not loaded
        batch_size = model.auto_tune_batch_size()

        assert batch_size == model._training_config.batch_size

    def test_estimate_memory_for_batch(self):
        """Test memory estimation for batch sizes."""
        model = Model()

        # Base memory + per_sample * batch_size
        # fp16: 4000 + 50 * batch_size
        memory_1 = model._estimate_memory_for_batch(1)
        memory_8 = model._estimate_memory_for_batch(8)
        memory_16 = model._estimate_memory_for_batch(16)

        assert memory_8 > memory_1
        assert memory_16 > memory_8

    def test_estimate_memory_fp32(self):
        """Test memory estimation with fp32."""
        model = Model()
        model.set_scalable_config(mixed_precision="fp32")

        # fp32 should use more memory than fp16
        memory_fp32 = model._estimate_memory_for_batch(4)

        model.set_scalable_config(mixed_precision="fp16")
        memory_fp16 = model._estimate_memory_for_batch(4)

        assert memory_fp32 > memory_fp16

    def test_estimate_memory_bf16(self):
        """Test memory estimation with bf16."""
        model = Model()
        model.set_scalable_config(mixed_precision="bf16")

        # bf16 should use same as fp16 (half of fp32)
        memory_bf16 = model._estimate_memory_for_batch(4)

        model.set_scalable_config(mixed_precision="fp16")
        memory_fp16 = model._estimate_memory_for_batch(4)

        assert memory_bf16 == memory_fp16


class TestScalableTrainingIntegration:
    """Integration tests for scalable training."""

    def test_model_with_all_scalable_features(self):
        """Test model with all scalable features enabled."""
        config = ScalableTrainingConfig(
            gradient_checkpointing=True,
            mixed_precision="bf16",
            batch_size_auto_tune=False,
            max_memory_mb=8192,
            num_workers=4,
            pin_memory=True,
            use_flash_attention=False,
        )
        model = Model(scalable_config=config)

        assert model._scalable_config.gradient_checkpointing is True
        assert model._scalable_config.mixed_precision == "bf16"
        assert model._scalable_config.max_memory_mb == 8192

    def test_scalable_config_persists(self):
        """Test that scalable config persists across method calls."""
        model = Model()
        model.set_scalable_config(
            gradient_checkpointing=False,
            mixed_precision="fp32",
            num_workers=16,
        )

        # Config should persist
        config = model.get_scalable_config()
        assert config.gradient_checkpointing is False
        assert config.mixed_precision == "fp32"
        assert config.num_workers == 16

    def test_training_config_independent(self):
        """Test that training config and scalable config are independent."""
        from autotrain import TrainingConfig
        
        model = Model()

        model.set_training_config(TrainingConfig(batch_size=8))
        model.set_scalable_config(num_workers=8)

        assert model._training_config.batch_size == 8
        assert model._scalable_config.num_workers == 8
        assert model._scalable_config.batch_size_auto_tune is False
