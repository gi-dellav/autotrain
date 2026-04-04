"""Splitter component - selects useful samples."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.model import Model
    from autotrain.data_types import Sample
    from autotrain.expert import Expert

from autotrain.components.config import ExpertWeight


class Splitter:
    """
    Splitter component - selects useful samples.

    Groups samples based on sample_multiplier and selects the most useful one.
    """

    def __init__(
        self,
        model: "Model",
        prompt: Union[str, list[str]],
        inference_config: "InferenceConfig",
        experts: Optional[list[Union["Expert", ExpertWeight]]] = None,
    ):
        self.model = model
        self.prompt = prompt
        self.inference_config = inference_config
        self._experts = []
        self._expert_weights: list[ExpertWeight] = []

        # Handle ExpertWeight objects or bare Experts
        if experts:
            for expert in experts:
                if isinstance(expert, ExpertWeight):
                    self._expert_weights.append(expert)
                    self._experts.append(expert.expert)
                else:
                    # Default weight = 1.0 for bare Expert
                    self._expert_weights.append(ExpertWeight(expert=expert, weight=1.0))
                    self._experts.append(expert)

    @property
    def experts(self) -> list["Expert"]:
        """Get list of experts."""
        return self._experts.copy()

    def add_expert(self, expert: "Expert", weight: float = 1.0) -> None:
        """Add an expert with a specific weight."""
        if expert not in self._experts:
            self._experts.append(expert)
            self._expert_weights.append(ExpertWeight(expert=expert, weight=weight))

    def remove_expert(self, expert: "Expert") -> None:
        """Remove an expert."""
        if expert in self._experts:
            self._experts.remove(expert)
            self._expert_weights = [ew for ew in self._expert_weights if ew.expert != expert]

    def _get_normalized_weights(self) -> list[tuple["Expert", float]]:
        """Get experts with normalized weights."""
        if not self._expert_weights:
            return []

        total_weight = sum(ew.weight for ew in self._expert_weights)
        if total_weight == 0:
            return []

        return [(ew.expert, ew.weight / total_weight) for ew in self._expert_weights]

    def _select_source(self) -> tuple[Optional["Expert"], bool]:
        """
        Select source using statistical weights.
        Returns tuple of (selected expert or None, is_expert)
        """
        if not self._expert_weights:
            return None, False

        # Calculate total expert weight
        total_expert_weight = sum(ew.weight for ew in self._expert_weights)

        # Model gets remaining weight (1.0 - total_expert_weight)
        model_weight = max(0.0, 1.0 - total_expert_weight)

        # Create weighted selection list
        selections: list[tuple[Optional["Expert"], float]] = []
        for ew in self._expert_weights:
            selections.append((ew.expert, ew.weight))
        selections.append((None, model_weight))

        total = sum(w for _, w in selections)
        if total == 0:
            return None, False

        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for expert, weight in selections:
            cumulative += weight / total
            if r <= cumulative:
                return expert, expert is not None

        return None, False

    def select(self, samples: list["Sample"], target_count: int, executor=None) -> list["Sample"]:
        """
        Select useful samples from the input (with optional parallelization).

        Args:
            samples: List of samples to select from
            target_count: Target number of samples to return
            executor: Optional shared ThreadPoolExecutor to use

        Returns:
            Selected samples
        """
        if len(samples) <= target_count:
            return samples

        # Group samples and select the best from each group
        sample_multiplier = self.model.sample_multiplier
        groups = [
            samples[i : i + sample_multiplier] for i in range(0, len(samples), sample_multiplier)
        ]

        # Separate trivial groups (size 1) from non-trivial
        selected = []
        non_trivial_groups = []

        for group in groups:
            if len(group) == 1:
                selected.append(group[0])
            else:
                non_trivial_groups.append(group)

        # Process non-trivial groups in parallel if configured
        if non_trivial_groups:
            use_async = getattr(self.model.inference_config, "use_async", False)
            max_workers = self.model.inference_config.max_workers

            if executor is not None or (use_async and max_workers > 1):
                try:
                    from concurrent.futures import ThreadPoolExecutor

                    if executor is not None:
                        # Use shared executor
                        best_from_groups = list(executor.map(self._select_best, non_trivial_groups))
                    else:
                        # Create own executor with configured workers
                        with ThreadPoolExecutor(max_workers=max_workers) as local_executor:
                            best_from_groups = list(
                                local_executor.map(self._select_best, non_trivial_groups)
                            )

                    selected.extend([s for s in best_from_groups if s is not None])
                except Exception as e:
                    print(f"Warning: Parallel selection failed, falling back to sequential: {e}")
                    for group in non_trivial_groups:
                        best = self._select_best(group)
                        if best is not None:
                            selected.append(best)
            else:
                # Sequential fallback
                for group in non_trivial_groups:
                    best = self._select_best(group)
                    if best is not None:
                        selected.append(best)

        # Ensure we return exactly target_count samples
        return selected[:target_count]

    def _select_best(self, samples: list["Sample"]) -> Optional["Sample"]:
        """
        Select the best sample from a group.

        Uses expert comparison when available, otherwise uses model-based selection.

        Args:
            samples: List of samples to select from

        Returns:
            Selected best sample
        """
        if len(samples) < 2:
            return samples[0] if samples else None

        # Select source using statistical weights
        expert, is_expert = self._select_source()

        if is_expert and expert:
            # Expert-based selection
            try:
                sample_outputs = [s.output_data for s in samples]

                # Use expert's compare method for pairwise comparison
                if len(sample_outputs) == 2:
                    comparison = expert.compare(
                        input_data=samples[0].input_data,
                        output_a=sample_outputs[0],
                        output_b=sample_outputs[1],
                    )
                    winner_idx = 0 if comparison["winner"] == "a" else 1
                    if comparison["winner"] == "tie":
                        winner_idx = 0  # Default to first on tie

                    selected = samples[winner_idx]
                    selected.metadata["selected_by"] = f"expert:{expert.model_name}"
                    selected.metadata["comparison_result"] = comparison
                    return selected
                else:
                    # Multiple samples: use expert's select method
                    selected_output = expert.select(sample_outputs)
                    for sample in samples:
                        if sample.output_data == selected_output:
                            sample.metadata["selected_by"] = f"expert:{expert.model_name}"
                            return sample
            except Exception as e:
                print(f"Warning: Expert selection failed, using model: {e}")

        # Model-based selection
        return self._model_select(samples)

    def _model_select(self, samples: list["Sample"]) -> Optional["Sample"]:
        """
        Select best sample using the in-training model.
        Uses model to compare samples pairwise.
        """
        if len(samples) < 2:
            return samples[0] if samples else None

        best_sample = samples[0]

        for i in range(1, len(samples)):
            comparison = self._compare_with_model(best_sample, samples[i])
            if comparison.get("winner") == "b":
                best_sample = samples[i]

        best_sample.metadata["selected_by"] = "model"
        best_sample.metadata["selection_method"] = "model_comparison"
        return best_sample

    def _compare_with_model(self, sample_a: "Sample", sample_b: "Sample") -> dict:
        """
        Use model to compare two samples.
        Returns comparison result with winner and explanation.
        """
        # Get splitter prompt
        prompt = self.model.prompts.get_splitter()

        # Format comparison prompt
        comparison_prompt = f"""{prompt}

Input: {sample_a.input_data}

Option A: {sample_a.output_data}
Option B: {sample_b.output_data}

Compare these options for training quality.
Select the better option and explain why.
Respond with 'A' or 'B' followed by your reasoning."""

        try:
            result = self.model.generate(
                prompt=comparison_prompt,
                temperature=0.3,
                max_tokens=128,
            )

            if isinstance(result, str):
                # Parse A/B choice from result
                if "A" in result.upper() and "B" not in result.upper():
                    return {"winner": "a", "explanation": result.strip()}
                elif "B" in result.upper():
                    return {"winner": "b", "explanation": result.strip()}
        except Exception as e:
            print(f"Warning: Model comparison failed: {e}")

        # Fallback: random choice if parsing fails
        return {"winner": random.choice(["a", "b"]), "explanation": "fallback_random"}

    def _score_sample(self, sample: "Sample") -> float:
        """
        Score a sample based on quality heuristics.

        Args:
            sample: Sample to score

        Returns:
            Quality score (higher = better)
        """
        output = sample.output_data
        score = 0.0

        # Length score (prefer moderate length, not too short or too long)
        length = len(output.split())
        if 20 <= length <= 200:
            score += 3.0
        elif 10 <= length < 20 or 200 < length <= 500:
            score += 2.0
        elif length > 500:
            score += 1.0  # Too long
        else:
            score += 0.5  # Too short

        # Completeness score (check for common completion indicators)
        if output and output[-1] in ".!?":
            score += 1.0  # Properly ended

        # Diversity score (check for varied vocabulary)
        words = output.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 2.0  # More diverse = better

        # Coherence indicators
        coherence_words = [
            "therefore",
            "because",
            "however",
            "thus",
            "hence",
            "first",
            "second",
            "finally",
            "in conclusion",
            "for example",
        ]
        for word in coherence_words:
            if word in output.lower():
                score += 0.5
                break

        return score

    async def select_async(
        self, samples: list["Sample"], target_count: int, executor=None
    ) -> list["Sample"]:
        """
        Select useful samples from the input (async).

        Args:
            samples: List of samples to select from
            target_count: Target number of samples to return
            executor: AsyncExecutor to use (required for async)

        Returns:
            Selected samples
        """
        if len(samples) <= target_count:
            return samples

        sample_multiplier = self.model.sample_multiplier
        groups = [
            samples[i : i + sample_multiplier] for i in range(0, len(samples), sample_multiplier)
        ]

        selected = []
        non_trivial_groups = []

        for group in groups:
            if len(group) == 1:
                selected.append(group[0])
            else:
                non_trivial_groups.append(group)

        if non_trivial_groups:
            if executor is None:
                raise ValueError("executor is required for async select")

            from functools import partial

            select_func = partial(self._select_best_async, executor=executor)
            best_from_groups = await executor.map_async(select_func, non_trivial_groups)
            selected.extend([s for s in best_from_groups if s is not None])

        return selected[:target_count]

    async def _select_best_async(
        self, samples: list["Sample"], executor=None
    ) -> Optional["Sample"]:
        """
        Select the best sample from a group (async).

        Uses expert comparison when available, otherwise uses model-based selection.

        Args:
            samples: List of samples to select from
            executor: AsyncExecutor for running sync model selection

        Returns:
            Selected best sample
        """
        if len(samples) < 2:
            return samples[0] if samples else None

        # Select source using statistical weights
        expert, is_expert = self._select_source()

        if is_expert and expert:
            # Expert-based selection
            try:
                sample_outputs = [s.output_data for s in samples]

                # Use expert's compare method for pairwise comparison
                if len(sample_outputs) == 2:
                    comparison = await expert.compare_async(
                        input_data=samples[0].input_data,
                        output_a=sample_outputs[0],
                        output_b=sample_outputs[1],
                    )
                    winner_idx = 0 if comparison["winner"] == "a" else 1
                    if comparison["winner"] == "tie":
                        winner_idx = 0  # Default to first on tie

                    selected = samples[winner_idx]
                    selected.metadata["selected_by"] = f"expert:{expert.model_name}"
                    selected.metadata["comparison_result"] = comparison
                    return selected
                else:
                    # Multiple samples: use expert's select method
                    selected_output = await expert.select_async(sample_outputs)
                    for sample in samples:
                        if sample.output_data == selected_output:
                            sample.metadata["selected_by"] = f"expert:{expert.model_name}"
                            return sample
            except Exception as e:
                print(f"Warning: Expert selection failed, using model: {e}")

        # Model-based selection (run sync _model_select in thread)
        if executor is None:
            raise ValueError("executor is required for model selection")
        return await executor.run_sync(self._model_select, samples)


__all__ = ["Splitter"]
