"""Benchmark evaluation logic."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from autotrain.benchmark.types import (
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSample,
    EvaluationMode,
)

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.model import Model


class Benchmark:
    """
    Benchmark for evaluating model quality during training.

    Contains a set of input-response pairs used to measure the quality
    of the fine-tuned model over iterations.
    """

    DEFAULT_THRESHOLDS = {
        EvaluationMode.EXACT_MATCH: 1.0,
        EvaluationMode.LLM_JUDGE: 0.7,
        EvaluationMode.BLEU: 0.3,
        EvaluationMode.ROUGE_L: 0.3,
        EvaluationMode.F1: 0.5,
    }

    def __init__(
        self,
        name: str = "default",
        expert_model_name: Optional[str] = None,
        expert_api_key: Optional[str] = None,
        expert_api_base: Optional[str] = None,
        thresholds: Optional[dict[EvaluationMode, float]] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize the Benchmark.

        Args:
            name: Name identifier for this benchmark
            expert_model_name: litellm model name for LLM judging
            expert_api_key: Optional API key for the expert model provider
            expert_api_base: Optional API base URL for the expert model
            thresholds: Custom thresholds for each evaluation mode (0-1)
            temperature: Temperature for LLM judge generation (0.0-2.0)
        """
        self.name = name
        self.expert_model_name = expert_model_name or "unsloth/Qwen3.5-27B-GGUF"
        self.expert_api_key = expert_api_key
        self.expert_api_base = expert_api_base
        self.temperature = temperature

        self._thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self._thresholds.update(thresholds)

        self._samples: list[BenchmarkSample] = []
        self._results_history: list[BenchmarkMetrics] = []
        self._best_metrics: Optional[BenchmarkMetrics] = None
        self._best_iteration: int = -1

    @property
    def thresholds(self) -> dict[EvaluationMode, float]:
        """Get the current thresholds."""
        return self._thresholds.copy()

    def set_threshold(self, mode: EvaluationMode, threshold: float) -> None:
        """Set threshold for a specific evaluation mode."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        self._thresholds[mode] = threshold

    def add_sample(
        self,
        input_data: str,
        expected_output: str,
        evaluation_mode: EvaluationMode = EvaluationMode.EXACT_MATCH,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a benchmark sample."""
        sample = BenchmarkSample(
            input_data=input_data,
            expected_output=expected_output,
            evaluation_mode=evaluation_mode,
            metadata=metadata or {},
        )
        self._samples.append(sample)

    def add_samples(self, samples: list[BenchmarkSample]) -> None:
        """Add multiple benchmark samples."""
        self._samples.extend(samples)

    def remove_sample(self, index: int) -> None:
        """Remove a sample by index."""
        if 0 <= index < len(self._samples):
            self._samples.pop(index)

    def clear_samples(self) -> None:
        """Clear all benchmark samples."""
        self._samples.clear()

    @property
    def samples(self) -> list[BenchmarkSample]:
        """Get all benchmark samples."""
        return self._samples.copy()

    @property
    def sample_count(self) -> int:
        """Get the number of benchmark samples."""
        return len(self._samples)

    @property
    def results_history(self) -> list[BenchmarkMetrics]:
        """Get the history of benchmark results."""
        return self._results_history.copy()

    @property
    def best_metrics(self) -> Optional[BenchmarkMetrics]:
        """Get the best benchmark metrics so far."""
        return self._best_metrics

    @property
    def best_iteration(self) -> int:
        """Get the iteration number with the best metrics."""
        return self._best_iteration

    def evaluate(
        self, model: "Model", iteration: int, inference_config: Optional["InferenceConfig"] = None
    ) -> BenchmarkMetrics:
        """Evaluate the model on all benchmark samples."""
        if not self._samples:
            raise ValueError("No benchmark samples to evaluate")

        model_outputs = self.generate_outputs(model, inference_config)
        return self.evaluate_outputs(model_outputs, iteration)

    def generate_outputs(
        self, model: "Model", inference_config: Optional["InferenceConfig"] = None
    ) -> list[tuple[BenchmarkSample, str]]:
        """Generate outputs for all benchmark samples. Returns list of (sample, output) tuples."""
        results = []
        for sample in self._samples:
            model_output = self._generate_output(model, sample.input_data, inference_config)
            results.append((sample, model_output))
        return results

    def evaluate_outputs(
        self, model_outputs: list[tuple[BenchmarkSample, str]], iteration: int
    ) -> BenchmarkMetrics:
        """Evaluate pre-generated model outputs against benchmark samples."""
        results: list[BenchmarkResult] = []
        correct_count = 0
        total_score = 0.0
        exact_match_count = 0
        llm_judge_count = 0
        errors = []

        for sample, model_output in model_outputs:
            threshold = self._thresholds.get(sample.evaluation_mode, 0.5)

            try:
                is_correct, score, details = self._evaluate(
                    model_output, sample.expected_output, sample.evaluation_mode, threshold
                )

                if sample.evaluation_mode == EvaluationMode.EXACT_MATCH:
                    exact_match_count += 1
                elif sample.evaluation_mode == EvaluationMode.LLM_JUDGE:
                    llm_judge_count += 1

            except Exception as e:
                errors.append(f"{sample.evaluation_mode.value}: {str(e)}")
                is_correct, score, details = (
                    False,
                    0.0,
                    {"error": str(e), "mode": sample.evaluation_mode.value},
                )

            result = BenchmarkResult(
                sample=sample,
                model_output=model_output,
                is_correct=is_correct,
                score=score,
                evaluation_details=details,
            )
            results.append(result)

            if is_correct:
                correct_count += 1
            total_score += score

        if errors:
            print(f"Warning: Evaluation errors occurred: {errors}")

        total_samples = len(model_outputs)
        metrics = BenchmarkMetrics(
            total_samples=total_samples,
            correct_samples=correct_count,
            accuracy=correct_count / total_samples if total_samples > 0 else 0.0,
            average_score=total_score / total_samples if total_samples > 0 else 0.0,
            exact_match_count=exact_match_count,
            llm_judge_count=llm_judge_count,
            bleu_count=sum(1 for s, _ in model_outputs if s.evaluation_mode == EvaluationMode.BLEU),
            rouge_l_count=sum(
                1 for s, _ in model_outputs if s.evaluation_mode == EvaluationMode.ROUGE_L
            ),
            f1_count=sum(1 for s, _ in model_outputs if s.evaluation_mode == EvaluationMode.F1),
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
        )

        self._update_best_metrics(metrics, iteration)
        self._results_history.append(metrics)

        return metrics

    def _evaluate(
        self, model_output: str, expected_output: str, mode: EvaluationMode, threshold: float
    ) -> tuple[bool, float, dict]:
        """Evaluate model output against expected output using the specified mode."""
        if mode == EvaluationMode.EXACT_MATCH:
            return self._evaluate_exact(model_output, expected_output, threshold)
        elif mode == EvaluationMode.LLM_JUDGE:
            return self._evaluate_llm_judge(model_output, expected_output, threshold)
        elif mode == EvaluationMode.BLEU:
            return self._evaluate_bleu(model_output, expected_output, threshold)
        elif mode == EvaluationMode.ROUGE_L:
            return self._evaluate_rouge_l(model_output, expected_output, threshold)
        elif mode == EvaluationMode.F1:
            return self._evaluate_f1(model_output, expected_output, threshold)
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

    def _generate_output(
        self, model: "Model", input_data: str, inference_config: Optional["InferenceConfig"] = None
    ) -> str:
        """Generate output from the model."""
        if hasattr(model, "_call_model") and callable(model._call_model):
            return model._call_model(input_data, inference_config)

        # Fallback: try to use unsloth model directly
        if model._fast_model is not None:
            try:
                inputs = model._tokenizer(input_data, return_tensors="pt").to(
                    model._fast_model.device
                )

                outputs = model._fast_model.generate(
                    **inputs,
                    max_new_tokens=inference_config.max_tokens if inference_config else 1024,
                    temperature=inference_config.temperature if inference_config else 0.7,
                    do_sample=inference_config.temperature > 0 if inference_config else True,
                    top_p=inference_config.top_p if inference_config else 0.9,
                )

                return model._tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Warning: Model generation failed: {e}")

        # Last resort: return placeholder
        return f"[Model output for: {input_data[:50]}...]"

    def _evaluate_exact(
        self, model_output: str, expected_output: str, threshold: float = 1.0
    ) -> tuple[bool, float, dict]:
        """Evaluate using exact match."""
        model_normalized = self._normalize_text(model_output)
        expected_normalized = self._normalize_text(expected_output)

        is_correct = model_normalized == expected_normalized

        score = self._calculate_similarity(model_normalized, expected_normalized)
        is_correct = is_correct or score >= threshold

        return (
            is_correct,
            score,
            {
                "mode": "exact_match",
                "threshold": threshold,
                "model_normalized": model_normalized,
                "expected_normalized": expected_normalized,
            },
        )

    def _evaluate_llm_judge(
        self, model_output: str, expected_output: str, threshold: float = 0.7
    ) -> tuple[bool, float, dict]:
        """Evaluate using an expert LLM as judge."""
        try:
            from litellm import completion
        except ImportError as e:
            raise ImportError(
                "litellm is required for LLM_JUDGE mode. Install with: pip install litellm"
            ) from e

        prompt = (
            f"Compare the following two responses and determine if they are "
            f"semantically equivalent.\n\n"
            f"Expected output:\n{expected_output}\n\n"
            f"Model output:\n{model_output}\n\n"
            f"Are these responses semantically equivalent (convey the same "
            f"meaning/information)?\n"
            f"Respond with a score from 0 to 10, where:\n"
            f"- 10 = Perfectly equivalent\n"
            f"- 7-9 = Mostly equivalent with minor differences\n"
            f"- 4-6 = Partially equivalent\n"
            f"- 1-3 = Slightly equivalent\n"
            f"- 0 = Not equivalent at all\n\n"
            f"Start your response with just the score number, then explain your reasoning."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator of text similarity and semantic equivalence."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = completion(
            model=self.expert_model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500,
            api_key=self.expert_api_key,
            api_base=self.expert_api_base,
        )

        response_text = response.choices[0].message.content

        score_match = re.search(r"\b([0-9]|10)\b", response_text)
        score = int(score_match.group()) if score_match else 5

        normalized_score = score / 10.0
        is_correct = normalized_score >= threshold

        return (
            is_correct,
            normalized_score,
            {
                "mode": "llm_judge",
                "expert_model": self.expert_model_name,
                "threshold": threshold,
                "raw_score": score,
                "explanation": response_text,
            },
        )

    def _evaluate_bleu(
        self, model_output: str, expected_output: str, threshold: float = 0.3
    ) -> tuple[bool, float, dict]:
        """Evaluate using BLEU score."""
        try:
            from datasets import load_metric
        except ImportError as e:
            raise ImportError(
                "datasets is required for BLEU evaluation. Install with: pip install datasets"
            ) from e

        bleu = load_metric("bleu")
        model_tokens = model_output.split()
        expected_tokens = [expected_output.split()]
        result = bleu.compute(predictions=[model_tokens], references=expected_tokens)
        bleu_score = result["bleu"]
        is_correct = bleu_score >= threshold
        return (
            is_correct,
            bleu_score,
            {"mode": "bleu", "threshold": threshold, "bleu_score": bleu_score},
        )

    def _evaluate_rouge_l(
        self, model_output: str, expected_output: str, threshold: float = 0.3
    ) -> tuple[bool, float, dict]:
        """Evaluate using ROUGE-L score."""
        try:
            from datasets import load_metric
        except ImportError as e:
            raise ImportError(
                "datasets is required for ROUGE evaluation. Install with: pip install datasets"
            ) from e

        rouge = load_metric("rouge")
        result = rouge.compute(predictions=[model_output], references=[expected_output])
        rouge_l = result["rougeL"].mid.fmeasure
        is_correct = rouge_l >= threshold
        return (
            is_correct,
            rouge_l,
            {"mode": "rouge_l", "threshold": threshold, "rouge_l": rouge_l},
        )

    def _evaluate_f1(
        self, model_output: str, expected_output: str, threshold: float = 0.5
    ) -> tuple[bool, float, dict]:
        """Evaluate using F1 score (word-level)."""
        model_tokens = set(model_output.lower().split())
        expected_tokens = set(expected_output.lower().split())

        if not model_tokens or not expected_tokens:
            return (
                False,
                0.0,
                {"mode": "f1", "threshold": threshold, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            )

        true_positives = len(model_tokens & expected_tokens)
        false_positives = len(model_tokens - expected_tokens)
        false_negatives = len(expected_tokens - model_tokens)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        is_correct = f1 >= threshold
        return (
            is_correct,
            f1,
            {
                "mode": "f1",
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove common formatting differences
        text = text.lower()
        return text

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)."""
        # Handle empty texts
        if not text1 and not text2:
            return 0.0

        if text1 == text2:
            return 1.0

        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _update_best_metrics(self, metrics: BenchmarkMetrics, iteration: int) -> None:
        """Update best metrics if current is better."""
        if self._best_metrics is None:
            self._best_metrics = metrics
            self._best_iteration = iteration
        elif metrics.accuracy > self._best_metrics.accuracy:
            self._best_metrics = metrics
            self._best_iteration = iteration
        elif metrics.accuracy == self._best_metrics.accuracy:
            # Tie-breaker: higher average score
            if metrics.average_score > self._best_metrics.average_score:
                self._best_metrics = metrics
                self._best_iteration = iteration

    def has_improved(self, threshold: float = 0.0, iterations_to_check: int = 1) -> bool:
        """Check if benchmark has improved over recent iterations."""
        if len(self._results_history) < 2:
            return False

        current = self._results_history[-1]

        if iterations_to_check >= len(self._results_history):
            previous = self._results_history[0]
        else:
            previous = self._results_history[-(iterations_to_check + 1)]

        return current.accuracy > previous.accuracy + threshold

    def has_stagnated(self, threshold: float = 0.01, iterations: int = 3) -> bool:
        """Check if benchmark has stagnated (no significant improvement)."""
        if len(self._results_history) < iterations + 1:
            return False

        recent = self._results_history[-iterations:]
        first_recent = recent[0]
        last_recent = recent[-1]

        improvement = last_recent.accuracy - first_recent.accuracy
        return improvement <= threshold

    def export(self, path: Union[str, Path], format: str = "json") -> None:
        """Export benchmark samples to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [sample.to_dict() for sample in self._samples]

        if format == "jsonl":
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        print(f"Benchmark '{self.name}' exported to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], format: str = "json") -> "Benchmark":
        """Load benchmark samples from a file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        if format == "jsonl":
            with open(path) as f:
                data = [json.loads(line) for line in f]
        else:
            with open(path) as f:
                data = json.load(f)

        benchmark = cls(name=path.stem)
        for item in data:
            benchmark.add_sample(
                input_data=item["input_data"],
                expected_output=item["expected_output"],
                evaluation_mode=EvaluationMode(item.get("evaluation_mode", "exact_match")),
                metadata=item.get("metadata", {}),
            )

        print(f"Loaded {len(benchmark._samples)} samples from {path}")
        return benchmark

    def export_results(self, path: Union[str, Path]) -> None:
        """Export benchmark results history to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "benchmark_name": self.name,
            "best_metrics": self._best_metrics.to_dict() if self._best_metrics else None,
            "best_iteration": self._best_iteration,
            "history": [m.to_dict() for m in self._results_history],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Benchmark results exported to {path}")


__all__ = ["Benchmark"]
