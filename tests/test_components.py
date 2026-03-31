"""Tests for pipeline components: Producer, Solver, Splitter, Reviewer, Checker."""

from unittest.mock import MagicMock

import pytest
from autotrain import InferenceConfig, Sample

from autotrain.components import (
    Checker,
    ExpertWeight,
    Producer,
    Reviewer,
    Solver,
    Splitter,
)
from autotrain.expert import Expert


class TestProducer:
    """Tests for Producer component."""

    def test_producer_init(self):
        """Test Producer initialization."""
        mock_model = MagicMock()
        config = InferenceConfig()

        producer = Producer(
            model=mock_model,
            prompt="Generate samples",
            inference_config=config,
        )

        assert producer.model == mock_model
        assert producer.prompt == "Generate samples"
        assert producer.inference_config == config

    def test_producer_generate(self):
        """Test Producer generates correct number of samples."""
        mock_model = MagicMock()
        config = InferenceConfig()

        producer = Producer(
            model=mock_model,
            prompt="Generate samples",
            inference_config=config,
        )

        samples = producer.generate(count=5)

        assert len(samples) == 5
        assert all(isinstance(s, Sample) for s in samples)
        assert all(s.metadata["source"] == "producer" for s in samples)

    def test_producer_sample_metadata(self):
        """Test Producer sample metadata."""
        mock_model = MagicMock()
        config = InferenceConfig()

        producer = Producer(
            model=mock_model,
            prompt="Generate samples",
            inference_config=config,
        )

        samples = producer.generate(count=3)

        for i, sample in enumerate(samples):
            assert sample.metadata["source"] == "producer"
            assert sample.metadata["iteration"] == 0
            # Input data should be a non-empty string (template-based fallback)
            assert isinstance(sample.input_data, str)
            assert len(sample.input_data) > 0
            assert sample.output_data == ""


class TestSolver:
    """Tests for Solver component."""

    def test_solver_init_no_experts(self):
        """Test Solver initialization without experts."""
        mock_model = MagicMock()
        config = InferenceConfig()

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
        )

        assert solver.model == mock_model
        assert solver.experts == []

    def test_solver_init_with_experts(self):
        """Test Solver initialization with experts."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
            experts=[expert],
        )

        assert len(solver.experts) == 1
        assert solver.experts[0] == expert

    def test_solver_add_expert(self):
        """Test adding expert to Solver."""
        mock_model = MagicMock()
        config = InferenceConfig()
        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
        )

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")
        solver.add_expert(expert, weight=0.5)

        assert len(solver.experts) == 1
        assert solver.experts[0] == expert

    def test_solver_remove_expert(self):
        """Test removing expert from Solver."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert1 = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")
        expert2 = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
            experts=[expert1, expert2],
        )

        solver.remove_expert(expert1)

        assert len(solver.experts) == 1
        assert expert1 not in solver.experts
        assert expert2 in solver.experts

    def test_solver_clear_experts(self):
        """Test clearing all experts from Solver."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
            experts=[expert],
        )

        solver.clear_experts()

        assert len(solver.experts) == 0

    def test_solver_solve_no_experts(self):
        """Test Solver.solve without experts uses model."""
        mock_model = MagicMock()
        config = InferenceConfig()

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
        )

        sample = Sample(input_data="What is 2+2?", output_data="")
        solved = solver.solve([sample])

        assert len(solved) == 1
        assert solved[0].output_data == "Solution for: What is 2+2?"
        assert solved[0].metadata["solver"] == "model"

    def test_solver_solve_with_expert(self, mock_litellm):
        """Test Solver.solve with expert uses expert."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
            experts=[expert],
        )

        # Force expert selection by mocking _select_source
        solver._select_source = MagicMock(return_value=(expert, True))

        sample = Sample(input_data="What is 2+2?", output_data="")
        solved = solver.solve([sample])

        assert len(solved) == 1
        # Expert should be called
        assert mock_litellm.called

    def test_solver_normalized_weights(self):
        """Test weight normalization in Solver."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert1 = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")
        expert2 = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        solver = Solver(
            model=mock_model,
            prompt="Solve problems",
            inference_config=config,
        )
        solver.add_expert(expert1, weight=0.5)
        solver.add_expert(expert2, weight=0.5)

        weights = solver._get_normalized_weights()

        assert len(weights) == 2
        # Weights should be normalized
        total = sum(w for _, w in weights)
        assert abs(total - 1.0) < 0.001


class TestSplitter:
    """Tests for Splitter component."""

    def test_splitter_init(self):
        """Test Splitter initialization."""
        mock_model = MagicMock()
        config = InferenceConfig()

        splitter = Splitter(
            model=mock_model,
            prompt="Select best",
            inference_config=config,
        )

        assert splitter.model == mock_model
        assert splitter.experts == []

    def test_splitter_add_expert(self):
        """Test adding expert to Splitter."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        splitter = Splitter(
            model=mock_model,
            prompt="Select best",
            inference_config=config,
        )

        splitter.add_expert(expert)

        assert len(splitter.experts) == 1

    def test_splitter_select_no_reduction(self):
        """Test Splitter.select when no reduction needed."""
        mock_model = MagicMock()
        mock_model.sample_multiplier = 2
        config = InferenceConfig()

        splitter = Splitter(
            model=mock_model,
            prompt="Select best",
            inference_config=config,
        )

        samples = [Sample(input_data=f"input_{i}", output_data=f"output_{i}") for i in range(3)]
        selected = splitter.select(samples, target_count=5)

        # Should return all samples when target > len(samples)
        assert len(selected) == 3

    def test_splitter_select_with_reduction(self):
        """Test Splitter.select with sample reduction."""
        mock_model = MagicMock()
        mock_model.sample_multiplier = 2
        config = InferenceConfig()

        splitter = Splitter(
            model=mock_model,
            prompt="Select best",
            inference_config=config,
        )

        # Create 6 samples, should select 3 (group by 2)
        samples = [Sample(input_data=f"input_{i}", output_data=f"output_{i}") for i in range(6)]
        selected = splitter.select(samples, target_count=3)

        assert len(selected) == 3

    def test_splitter_select_best_default(self):
        """Test Splitter._select_best uses default when no expert."""
        mock_model = MagicMock()
        config = InferenceConfig()

        splitter = Splitter(
            model=mock_model,
            prompt="Select best",
            inference_config=config,
        )

        samples = [
            Sample(input_data="input", output_data="output1"),
            Sample(input_data="input", output_data="output2"),
        ]

        best = splitter._select_best(samples)

        assert best == samples[0]
        assert best.metadata["selected_by"] == "default"


class TestReviewer:
    """Tests for Reviewer component."""

    def test_reviewer_init(self):
        """Test Reviewer initialization."""
        mock_model = MagicMock()
        config = InferenceConfig()

        reviewer = Reviewer(
            model=mock_model,
            prompt="Review samples",
            inference_config=config,
        )

        assert reviewer.model == mock_model
        assert reviewer.experts == []

    def test_reviewer_add_expert(self):
        """Test adding expert to Reviewer."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        reviewer = Reviewer(
            model=mock_model,
            prompt="Review samples",
            inference_config=config,
        )

        reviewer.add_expert(expert)

        assert len(reviewer.experts) == 1

    def test_reviewer_review_no_expert(self):
        """Test Reviewer.review without expert."""
        mock_model = MagicMock()
        config = InferenceConfig()

        reviewer = Reviewer(
            model=mock_model,
            prompt="Review samples",
            inference_config=config,
        )

        sample = Sample(input_data="input", output_data="output")
        reviewed = reviewer.review([sample])

        assert len(reviewed) == 1
        assert reviewed[0].metadata["review"]["reviewer"] == "none"
        assert reviewed[0].metadata["review"]["score"] == 5

    def test_reviewer_review_with_expert(self, mock_litellm):
        """Test Reviewer.review with expert."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")

        reviewer = Reviewer(
            model=mock_model,
            prompt="Review samples",
            inference_config=config,
            experts=[expert],
        )

        sample = Sample(input_data="input", output_data="output")
        reviewed = reviewer.review([sample])

        assert len(reviewed) == 1
        assert "review" in reviewed[0].metadata
        # Expert should be called
        assert mock_litellm.called


class TestChecker:
    """Tests for Checker component."""

    def test_checker_init(self):
        """Test Checker initialization."""
        mock_model = MagicMock()
        config = InferenceConfig()

        checker = Checker(
            model=mock_model,
            prompt="Check correctness",
            inference_config=config,
        )

        assert checker.model == mock_model
        assert checker.experts == []

    def test_checker_add_expert(self):
        """Test adding expert to Checker."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        checker = Checker(
            model=mock_model,
            prompt="Check correctness",
            inference_config=config,
        )

        checker.add_expert(expert)

        assert len(checker.experts) == 1

    def test_checker_verify_no_expert(self):
        """Test Checker.verify without expert passes all."""
        mock_model = MagicMock()
        config = InferenceConfig()

        checker = Checker(
            model=mock_model,
            prompt="Check correctness",
            inference_config=config,
        )

        samples = [
            Sample(input_data="input1", output_data="output1"),
            Sample(input_data="input2", output_data="output2"),
        ]
        verified = checker.verify(samples)

        # Without expert, all samples pass
        assert len(verified) == 2
        assert all(s.metadata["check"]["is_correct"] for s in verified)

    def test_checker_verify_with_expert(self, mock_litellm):
        """Test Checker.verify with expert."""
        mock_model = MagicMock()
        config = InferenceConfig()
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")

        checker = Checker(
            model=mock_model,
            prompt="Check correctness",
            inference_config=config,
            experts=[expert],
        )

        sample = Sample(input_data="input", output_data="output")
        checker.verify([sample])

        # Expert should be called
        assert mock_litellm.called


class TestExpertWeight:
    """Tests for ExpertWeight dataclass."""

    def test_expert_weight_valid(self):
        """Test ExpertWeight with valid weight."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")
        weight = ExpertWeight(expert=expert, weight=0.5)

        assert weight.expert == expert
        assert weight.weight == 0.5

    def test_expert_weight_zero(self):
        """Test ExpertWeight with zero weight."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")
        weight = ExpertWeight(expert=expert, weight=0.0)

        assert weight.weight == 0.0

    def test_expert_weight_negative_raises(self):
        """Test ExpertWeight with negative weight raises error."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF")

        with pytest.raises(ValueError, match="must be non-negative"):
            ExpertWeight(expert=expert, weight=-0.5)
