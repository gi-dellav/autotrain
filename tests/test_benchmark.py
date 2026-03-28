"""Tests for Benchmark class."""

import json
from unittest.mock import MagicMock

import pytest

from autotrain.benchmark import (
    Benchmark,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSample,
    EvaluationMode,
)


class TestEvaluationMode:
    """Tests for EvaluationMode enum."""

    def test_evaluation_mode_values(self):
        """Test EvaluationMode enum values."""
        assert EvaluationMode.EXACT_MATCH.value == "exact_match"
        assert EvaluationMode.LLM_JUDGE.value == "llm_judge"


class TestBenchmarkSample:
    """Tests for BenchmarkSample dataclass."""

    def test_sample_creation(self):
        """Test BenchmarkSample creation."""
        sample = BenchmarkSample(
            input_data="What is 2+2?",
            expected_output="4",
            evaluation_mode=EvaluationMode.EXACT_MATCH,
        )

        assert sample.input_data == "What is 2+2?"
        assert sample.expected_output == "4"
        assert sample.evaluation_mode == EvaluationMode.EXACT_MATCH

    def test_sample_to_dict(self):
        """Test BenchmarkSample.to_dict."""
        sample = BenchmarkSample(
            input_data="test input",
            expected_output="test output",
            evaluation_mode=EvaluationMode.EXACT_MATCH,
            metadata={"key": "value"},
        )

        data = sample.to_dict()

        assert data["input_data"] == "test input"
        assert data["expected_output"] == "test output"
        assert data["evaluation_mode"] == "exact_match"
        assert data["metadata"]["key"] == "value"

    def test_sample_from_dict(self):
        """Test BenchmarkSample.from_dict."""
        data = {
            "input_data": "test input",
            "expected_output": "test output",
            "evaluation_mode": "exact_match",
            "metadata": {"key": "value"},
        }

        sample = BenchmarkSample.from_dict(data)

        assert sample.input_data == "test input"
        assert sample.expected_output == "test output"
        assert sample.evaluation_mode == EvaluationMode.EXACT_MATCH

    def test_sample_default_mode(self):
        """Test BenchmarkSample default evaluation mode."""
        sample = BenchmarkSample(
            input_data="input",
            expected_output="output",
        )

        assert sample.evaluation_mode == EvaluationMode.EXACT_MATCH


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test BenchmarkResult creation."""
        sample = BenchmarkSample(input_data="input", expected_output="output")
        result = BenchmarkResult(
            sample=sample,
            model_output="model output",
            is_correct=True,
            score=0.9,
        )

        assert result.is_correct is True
        assert result.score == 0.9

    def test_result_to_dict(self):
        """Test BenchmarkResult.to_dict."""
        sample = BenchmarkSample(input_data="input", expected_output="output")
        result = BenchmarkResult(
            sample=sample,
            model_output="model output",
            is_correct=True,
            score=1.0,
        )

        data = result.to_dict()

        assert data["is_correct"] is True
        assert data["score"] == 1.0
        assert data["model_output"] == "model output"


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_metrics_default(self):
        """Test BenchmarkMetrics default values."""
        metrics = BenchmarkMetrics()

        assert metrics.total_samples == 0
        assert metrics.correct_samples == 0
        assert metrics.accuracy == 0.0
        assert metrics.average_score == 0.0

    def test_metrics_calculation(self):
        """Test BenchmarkMetrics with values."""
        metrics = BenchmarkMetrics(
            total_samples=10,
            correct_samples=8,
            accuracy=0.8,
            average_score=0.85,
        )

        assert metrics.total_samples == 10
        assert metrics.correct_samples == 8
        assert metrics.accuracy == 0.8

    def test_metrics_to_dict(self):
        """Test BenchmarkMetrics.to_dict."""
        metrics = BenchmarkMetrics(
            total_samples=10,
            correct_samples=8,
            accuracy=0.8,
            iteration=5,
        )

        data = metrics.to_dict()

        assert data["total_samples"] == 10
        assert data["accuracy"] == 0.8
        assert data["iteration"] == 5


class TestBenchmark:
    """Tests for Benchmark class."""

    def test_benchmark_init_default(self):
        """Test Benchmark initialization with defaults."""
        benchmark = Benchmark()

        assert benchmark.name == "default"
        assert benchmark.expert_model_name == "gpt-4"
        assert benchmark.sample_count == 0
        assert benchmark.best_metrics is None

    def test_benchmark_init_custom(self):
        """Test Benchmark initialization with custom params."""
        benchmark = Benchmark(
            name="math_benchmark",
            expert_model_name="claude-3",
            expert_api_key="test-key",
        )

        assert benchmark.name == "math_benchmark"
        assert benchmark.expert_model_name == "claude-3"
        assert benchmark.expert_api_key == "test-key"

    def test_add_sample(self):
        """Test adding benchmark sample."""
        benchmark = Benchmark()
        benchmark.add_sample(
            input_data="What is 2+2?",
            expected_output="4",
        )

        assert benchmark.sample_count == 1

    def test_add_sample_with_mode(self):
        """Test adding benchmark sample with evaluation mode."""
        benchmark = Benchmark()
        benchmark.add_sample(
            input_data="Explain gravity",
            expected_output="Gravity is a force...",
            evaluation_mode=EvaluationMode.LLM_JUDGE,
        )

        assert benchmark.samples[0].evaluation_mode == EvaluationMode.LLM_JUDGE

    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        benchmark = Benchmark()
        samples = [
            BenchmarkSample(input_data=f"input_{i}", expected_output=f"output_{i}")
            for i in range(5)
        ]

        benchmark.add_samples(samples)

        assert benchmark.sample_count == 5

    def test_remove_sample(self):
        """Test removing a sample."""
        benchmark = Benchmark()
        benchmark.add_sample("input1", "output1")
        benchmark.add_sample("input2", "output2")
        benchmark.add_sample("input3", "output3")

        benchmark.remove_sample(1)

        assert benchmark.sample_count == 2

    def test_remove_sample_invalid_index(self):
        """Test removing sample with invalid index."""
        benchmark = Benchmark()
        benchmark.add_sample("input", "output")

        # Should not raise, just do nothing
        benchmark.remove_sample(100)
        assert benchmark.sample_count == 1

    def test_clear_samples(self):
        """Test clearing all samples."""
        benchmark = Benchmark()
        benchmark.add_sample("input1", "output1")
        benchmark.add_sample("input2", "output2")

        benchmark.clear_samples()

        assert benchmark.sample_count == 0

    def test_samples_returns_copy(self):
        """Test that samples property returns a copy."""
        benchmark = Benchmark()
        benchmark.add_sample("input", "output")

        samples = benchmark.samples
        samples.clear()

        # Original should be unchanged
        assert benchmark.sample_count == 1

    def test_evaluate_exact_match(self):
        """Test benchmark evaluation with exact match."""
        benchmark = Benchmark()
        benchmark.add_sample(
            input_data="What is 2+2?",
            expected_output="4",
            evaluation_mode=EvaluationMode.EXACT_MATCH,
        )

        # Mock model
        mock_model = MagicMock()
        mock_model._call_model = MagicMock(return_value="4")

        metrics = benchmark.evaluate(mock_model, iteration=1)

        assert metrics.total_samples == 1
        assert metrics.accuracy == 1.0
        assert metrics.correct_samples == 1

    def test_evaluate_exact_match_incorrect(self):
        """Test benchmark evaluation with incorrect answer."""
        benchmark = Benchmark()
        benchmark.add_sample(
            input_data="What is 2+2?",
            expected_output="4",
            evaluation_mode=EvaluationMode.EXACT_MATCH,
        )

        mock_model = MagicMock()
        mock_model._call_model = MagicMock(return_value="5")

        metrics = benchmark.evaluate(mock_model, iteration=1)

        assert metrics.accuracy == 0.0
        assert metrics.correct_samples == 0

    def test_evaluate_updates_best_metrics(self):
        """Test that evaluate updates best metrics."""
        benchmark = Benchmark()
        benchmark.add_sample("input", "output")

        mock_model = MagicMock()
        mock_model._call_model = MagicMock(return_value="output")

        benchmark.evaluate(mock_model, iteration=1)

        assert benchmark.best_metrics is not None
        assert benchmark.best_iteration == 1

    def test_evaluate_stores_history(self):
        """Test that evaluate stores results history."""
        benchmark = Benchmark()
        benchmark.add_sample("input", "output")

        mock_model = MagicMock()
        mock_model._call_model = MagicMock(return_value="output")

        benchmark.evaluate(mock_model, iteration=1)
        benchmark.evaluate(mock_model, iteration=2)

        assert len(benchmark.results_history) == 2

    def test_has_improved_true(self):
        """Test has_improved returns True when improved."""
        benchmark = Benchmark()
        benchmark.add_sample("input", "output")

        # Add results with different accuracies
        metrics1 = BenchmarkMetrics(total_samples=1, correct_samples=0, accuracy=0.5)
        metrics2 = BenchmarkMetrics(total_samples=1, correct_samples=1, accuracy=1.0)

        benchmark._results_history = [metrics1, metrics2]

        assert benchmark.has_improved() is True

    def test_has_improved_false(self):
        """Test has_improved returns False when not improved."""
        benchmark = Benchmark()

        metrics1 = BenchmarkMetrics(total_samples=1, correct_samples=1, accuracy=1.0)
        metrics2 = BenchmarkMetrics(total_samples=1, correct_samples=1, accuracy=1.0)

        benchmark._results_history = [metrics1, metrics2]

        assert benchmark.has_improved(threshold=0.01) is False

    def test_has_improved_no_history(self):
        """Test has_improved with insufficient history."""
        benchmark = Benchmark()

        # Only one result, can't compare
        metrics = BenchmarkMetrics(accuracy=1.0)
        benchmark._results_history = [metrics]

        assert benchmark.has_improved() is False

    def test_has_stagnated_true(self):
        """Test has_stagnated returns True when stagnated."""
        benchmark = Benchmark()

        # Add results with no improvement
        metrics = [
            BenchmarkMetrics(accuracy=0.5),
            BenchmarkMetrics(accuracy=0.51),
            BenchmarkMetrics(accuracy=0.50),
            BenchmarkMetrics(accuracy=0.51),
        ]

        benchmark._results_history = metrics

        assert benchmark.has_stagnated(threshold=0.05, iterations=3) is True

    def test_has_stagnated_false(self):
        """Test has_stagnated returns False when improving."""
        benchmark = Benchmark()

        # Add results with significant improvement
        metrics = [
            BenchmarkMetrics(accuracy=0.5),
            BenchmarkMetrics(accuracy=0.6),
            BenchmarkMetrics(accuracy=0.7),
            BenchmarkMetrics(accuracy=0.8),
        ]

        benchmark._results_history = metrics

        assert benchmark.has_stagnated(threshold=0.05, iterations=3) is False

    def test_has_stagnated_insufficient_history(self):
        """Test has_stagnated with insufficient history."""
        benchmark = Benchmark()

        # Only 2 results, need at least 4 for iterations=3
        metrics = [
            BenchmarkMetrics(accuracy=0.5),
            BenchmarkMetrics(accuracy=0.5),
        ]

        benchmark._results_history = metrics

        assert benchmark.has_stagnated(iterations=3) is False

    def test_export_json(self, temp_dir):
        """Test exporting benchmark to JSON."""
        benchmark = Benchmark(name="test")
        benchmark.add_sample("input1", "output1")
        benchmark.add_sample("input2", "output2")

        output_path = temp_dir / "benchmark.json"
        benchmark.export(str(output_path), format="json")

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 2

    def test_export_jsonl(self, temp_dir):
        """Test exporting benchmark to JSONL."""
        benchmark = Benchmark(name="test")
        benchmark.add_sample("input1", "output1")
        benchmark.add_sample("input2", "output2")

        output_path = temp_dir / "benchmark.jsonl"
        benchmark.export(str(output_path), format="jsonl")

        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_load_json(self, temp_dir):
        """Test loading benchmark from JSON."""
        # Create test file
        data = [
            {"input_data": "input1", "expected_output": "output1"},
            {"input_data": "input2", "expected_output": "output2"},
        ]

        input_path = temp_dir / "benchmark.json"
        with open(input_path, "w") as f:
            json.dump(data, f)

        benchmark = Benchmark.load(str(input_path), format="json")

        assert benchmark.sample_count == 2

    def test_load_jsonl(self, temp_dir):
        """Test loading benchmark from JSONL."""
        input_path = temp_dir / "benchmark.jsonl"
        with open(input_path, "w") as f:
            f.write('{"input_data": "input1", "expected_output": "output1"}\n')
            f.write('{"input_data": "input2", "expected_output": "output2"}\n')

        benchmark = Benchmark.load(str(input_path), format="jsonl")

        assert benchmark.sample_count == 2

    def test_load_not_found(self):
        """Test loading benchmark from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Benchmark.load("/nonexistent/path.json")

    def test_export_results(self, temp_dir):
        """Test exporting benchmark results."""
        benchmark = Benchmark(name="test")
        benchmark.add_sample("input", "output")

        # Add some metrics
        metrics = BenchmarkMetrics(
            total_samples=1,
            correct_samples=1,
            accuracy=1.0,
            iteration=1,
        )
        benchmark._results_history = [metrics]
        benchmark._best_metrics = metrics
        benchmark._best_iteration = 1

        output_path = temp_dir / "results.json"
        benchmark.export_results(str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["benchmark_name"] == "test"
        assert data["best_iteration"] == 1
        assert len(data["history"]) == 1

    def test_normalize_text(self):
        """Test text normalization."""
        benchmark = Benchmark()

        text1 = "  Hello   World  "
        text2 = "hello world"

        norm1 = benchmark._normalize_text(text1)
        norm2 = benchmark._normalize_text(text2)

        assert norm1 == norm2

    def test_calculate_similarity_identical(self):
        """Test similarity calculation for identical texts."""
        benchmark = Benchmark()

        similarity = benchmark._calculate_similarity("hello world", "hello world")

        assert similarity == 1.0

    def test_calculate_similarity_different(self):
        """Test similarity calculation for different texts."""
        benchmark = Benchmark()

        similarity = benchmark._calculate_similarity("hello world", "foo bar")

        assert similarity < 1.0

    def test_calculate_similarity_empty(self):
        """Test similarity calculation for empty texts."""
        benchmark = Benchmark()

        similarity = benchmark._calculate_similarity("", "")

        assert similarity == 0.0

    def test_update_best_metrics_first(self):
        """Test updating best metrics (first)."""
        benchmark = Benchmark()

        metrics = BenchmarkMetrics(accuracy=0.8, iteration=1)
        benchmark._update_best_metrics(metrics, iteration=1)

        assert benchmark._best_metrics == metrics
        assert benchmark._best_iteration == 1

    def test_update_best_metrics_better(self):
        """Test updating best metrics (better)."""
        benchmark = Benchmark()

        metrics1 = BenchmarkMetrics(accuracy=0.7, iteration=1)
        metrics2 = BenchmarkMetrics(accuracy=0.9, iteration=2)

        benchmark._update_best_metrics(metrics1, iteration=1)
        benchmark._update_best_metrics(metrics2, iteration=2)

        assert benchmark._best_metrics.accuracy == 0.9
        assert benchmark._best_iteration == 2

    def test_update_best_metrics_worse(self):
        """Test updating best metrics (worse - should not update)."""
        benchmark = Benchmark()

        metrics1 = BenchmarkMetrics(accuracy=0.9, iteration=1)
        metrics2 = BenchmarkMetrics(accuracy=0.7, iteration=2)

        benchmark._update_best_metrics(metrics1, iteration=1)
        benchmark._update_best_metrics(metrics2, iteration=2)

        assert benchmark._best_metrics.accuracy == 0.9
        assert benchmark._best_iteration == 1

    def test_update_best_metrics_tie_breaker(self):
        """Test updating best metrics with tie-breaker."""
        benchmark = Benchmark()

        metrics1 = BenchmarkMetrics(accuracy=0.8, average_score=0.75, iteration=1)
        metrics2 = BenchmarkMetrics(accuracy=0.8, average_score=0.85, iteration=2)

        benchmark._update_best_metrics(metrics1, iteration=1)
        benchmark._update_best_metrics(metrics2, iteration=2)

        # Should update due to higher average_score
        assert benchmark._best_iteration == 2
