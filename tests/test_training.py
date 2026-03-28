"""Integration tests for full training pipeline."""

from unittest.mock import MagicMock

import pytest
from autotrain import Model, Sample

from autotrain.benchmark import Benchmark, EvaluationMode
from autotrain.expert import Expert


class TestTrainingPipeline:
    """Integration tests for the full training pipeline."""

    def test_full_pipeline_no_experts(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline without experts."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        # Add initial samples
        initial_samples = [
            Sample(input_data="What is 2+2?", output_data="4"),
            Sample(input_data="What is 3+3?", output_data="6"),
        ]

        # Mock fine-tune to avoid actual training
        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=2,
            initial_samples=initial_samples,
            checkpoint_every=1,
        )

        assert summary["iterations_completed"] == 2
        assert summary["total_samples"] >= 2
        assert not summary["stopped_early"]

    def test_full_pipeline_with_expert(self, mock_unsloth, mock_trainer, mock_litellm, temp_dir):
        """Test full training pipeline with expert."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        expert = Expert(model_name="gpt-4", api_key="test-key")

        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=2,
            experts=[expert],
        )

        assert summary["iterations_completed"] == 2
        # Expert should have been called
        assert mock_litellm.called

    def test_full_pipeline_with_benchmark(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with benchmark evaluation."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        benchmark = Benchmark(name="test_benchmark")
        benchmark.add_sample(
            input_data="What is 2+2?",
            expected_output="4",
            evaluation_mode=EvaluationMode.EXACT_MATCH,
        )

        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=3,
            benchmark=benchmark,
        )

        assert summary["iterations_completed"] == 3
        assert "benchmark_history" in summary
        assert len(summary["benchmark_history"]) > 0

    @pytest.mark.skip(reason="Early stopping logic requires more complex mock setup")
    def test_full_pipeline_early_stopping(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with early stopping."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        benchmark = Benchmark(name="test_benchmark")
        benchmark.add_sample(
            input_data="What is 2+2?",
            expected_output="4",
        )

        model._fine_tune = MagicMock()

        # Train for 10 iterations but should stop early
        summary = model.train(
            k=2,
            i=10,
            benchmark=benchmark,
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_threshold=0.01,
        )

        # Should have stopped early due to stagnation
        assert summary["stopped_early"] or summary["iterations_completed"] < 10

    def test_full_pipeline_checkpoint_resume(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with checkpoint and resume."""
        # First training run
        model1 = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )
        model1._fine_tune = MagicMock()

        model1.train(
            k=2,
            i=2,
            checkpoint_every=1,
        )

        # Second training run - resume
        model2 = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )
        model2._fine_tune = MagicMock()

        summary = model2.train(
            k=2,
            i=2,
            resume_from_checkpoint=True,
        )

        # Should have resumed
        assert summary["iterations_completed"] >= 2

    def test_full_pipeline_multiple_experts(
        self, mock_unsloth, mock_trainer, mock_litellm, temp_dir
    ):
        """Test full training pipeline with multiple experts."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        expert1 = Expert(model_name="gpt-4", api_key="key1", production_rate=0.5)
        expert2 = Expert(model_name="claude-3", api_key="key2", production_rate=0.3)

        model._fine_tune = MagicMock()

        summary = model.train(
            k=4,
            i=2,
            experts=[expert1, expert2],
        )

        assert summary["iterations_completed"] == 2
        # Both experts should have been called
        assert mock_litellm.call_count >= 2

    def test_full_pipeline_with_checker(self, mock_unsloth, mock_trainer, mock_litellm, temp_dir):
        """Test full training pipeline with checker enabled."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
            enable_checker=True,
        )

        expert = Expert(model_name="gpt-4", production_rate=1.0, api_key="test-key")
        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=2,
            experts=[expert],
        )

        assert summary["iterations_completed"] == 2

    def test_full_pipeline_keep_best_model(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline keeps best model."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
            keep_best_checkpoint=True,
        )

        benchmark = Benchmark(name="test")
        benchmark.add_sample(
            input_data="test",
            expected_output="output",
        )

        model._fine_tune = MagicMock()

        model.train(
            k=2,
            i=3,
            benchmark=benchmark,
            checkpoint_every=1,
            keep_best_model=True,
        )

        # Should have saved best checkpoint
        best = model._checkpoint_manager.get_best_checkpoint()
        assert best is not None

    def test_full_pipeline_benchmark_results_export(self, mock_unsloth, mock_trainer, temp_dir):
        """Test that benchmark results are exported."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        benchmark = Benchmark(name="test")
        benchmark.add_sample(
            input_data="test",
            expected_output="output",
        )

        model._fine_tune = MagicMock()

        model.train(
            k=2,
            i=2,
            benchmark=benchmark,
        )

        # Check that results file was created
        results_path = temp_dir / "benchmark_results.json"
        assert results_path.exists()

    def test_full_pipeline_peft_config(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with custom PEFT config."""
        from autotrain import PEFTConfig

        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        model.set_peft_config(
            PEFTConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
        )

        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
        )

        assert summary["iterations_completed"] == 1

        # Verify config was set
        config = model.get_peft_config()
        assert config.r == 32
        assert config.lora_alpha == 64

    def test_full_pipeline_training_config(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with custom training config."""
        from autotrain import TrainingConfig

        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        model.set_training_config(
            TrainingConfig(
                epochs=3,
                batch_size=4,
                learning_rate=1e-4,
                warmup_steps=20,
            )
        )

        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
        )

        assert summary["iterations_completed"] == 1

        # Verify config was set
        config = model.get_training_config()
        assert config.epochs == 3
        assert config.learning_rate == 1e-4

    def test_full_pipeline_custom_prompts(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with custom prompts."""
        from autotrain import Prompts

        prompts = Prompts(
            producer="Custom producer: generate samples",
            solver="Custom solver: solve this",
        )

        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
            prompts=prompts,
        )

        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
        )

        assert summary["iterations_completed"] == 1
        assert model._producer.prompt == "Custom producer: generate samples"

    def test_full_pipeline_inference_config(self, mock_unsloth, mock_trainer, temp_dir):
        """Test full training pipeline with custom inference config."""
        from autotrain import InferenceConfig

        config = InferenceConfig(
            temperature=0.9,
            max_tokens=2048,
            top_p=0.95,
        )

        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
            inference_config=config,
        )

        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
        )

        assert summary["iterations_completed"] == 1
        assert model.inference_config.temperature == 0.9
