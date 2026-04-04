"""Benchmark data classes and types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class EvaluationMode(Enum):
    """Evaluation mode for benchmark samples."""

    EXACT_MATCH = "exact_match"
    LLM_JUDGE = "llm_judge"
    BLEU = "bleu"
    ROUGE_L = "rouge_l"
    F1 = "f1"


@dataclass
class BenchmarkSample:
    """A single benchmark sample with input and expected output."""

    input_data: str
    expected_output: str
    evaluation_mode: EvaluationMode = EvaluationMode.EXACT_MATCH
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "evaluation_mode": self.evaluation_mode.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        """Create from dictionary."""
        return cls(
            input_data=data["input_data"],
            expected_output=data["expected_output"],
            evaluation_mode=EvaluationMode(data.get("evaluation_mode", "exact_match")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BenchmarkResult:
    """Result of evaluating a single benchmark sample."""

    sample: BenchmarkSample
    model_output: str
    is_correct: bool
    score: float = 0.0
    evaluation_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_data": self.sample.input_data,
            "expected_output": self.sample.expected_output,
            "model_output": self.model_output,
            "is_correct": self.is_correct,
            "score": self.score,
            "evaluation_mode": self.sample.evaluation_mode.value,
            "evaluation_details": self.evaluation_details,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from a benchmark run."""

    total_samples: int = 0
    correct_samples: int = 0
    accuracy: float = 0.0
    average_score: float = 0.0
    exact_match_count: int = 0
    llm_judge_count: int = 0
    bleu_count: int = 0
    rouge_l_count: int = 0
    f1_count: int = 0
    iteration: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "accuracy": self.accuracy,
            "average_score": self.average_score,
            "exact_match_count": self.exact_match_count,
            "llm_judge_count": self.llm_judge_count,
            "bleu_count": self.bleu_count,
            "rouge_l_count": self.rouge_l_count,
            "f1_count": self.f1_count,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
        }


__all__ = ["EvaluationMode", "BenchmarkSample", "BenchmarkResult", "BenchmarkMetrics"]
