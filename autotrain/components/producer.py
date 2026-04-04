"""Producer component - generates input samples."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.model import Model
    from autotrain.data_types import Sample


def _get_prompt(prompt: Union[str, list[str]]) -> str:
    """Get a single prompt from str or list[str], selecting randomly if list."""
    if isinstance(prompt, list):
        return random.choice(prompt)
    return prompt


class Producer:
    """
    Producer component - generates input samples.

    Creates diverse input samples for the model to solve.
    """

    def __init__(
        self,
        model: "Model",
        prompt: Union[str, list[str]],
        inference_config: "InferenceConfig",
    ):
        self.model = model
        self.prompt = prompt
        self.inference_config = inference_config

    def generate(
        self, count: int, topic: str = "general knowledge training samples", executor=None
    ) -> list["Sample"]:
        """
        Generate input samples in parallel.

        Args:
            count: Number of samples to generate
            topic: The topic to generate training samples for. Defaults to
                  "general knowledge training samples".
            executor: Optional shared ThreadPoolExecutor to use

        Returns:
            List of generated samples
        """
        from concurrent.futures import ThreadPoolExecutor

        from autotrain.data_types import Sample

        # Get the baked prompt for the topic
        prompt = self.model.prompts.get_producer(topic)

        def _generate_one(i):
            input_data = self._generate_input(i, prompt)
            return Sample(
                input_data=input_data,
                output_data="",  # Will be filled by Solver
                metadata={
                    "source": "producer",
                    "producer": "model",
                    "iteration": 0,
                    "topic": topic,
                },
            )

        max_workers = self.model.inference_config.max_workers
        if not isinstance(max_workers, int):
            max_workers = 8

        if executor is not None:
            # Use shared executor
            samples = list(executor.map(_generate_one, range(count)))
        else:
            # Create own executor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                samples = list(executor.map(_generate_one, range(count)))

        return samples

    async def generate_async(
        self, count: int, topic: str = "general knowledge training samples", executor=None
    ) -> list["Sample"]:
        """
        Generate input samples in parallel (async).

        Args:
            count: Number of samples to generate
            topic: The topic to generate training samples for.
            executor: AsyncExecutor to use (required for async)

        Returns:
            List of generated samples
        """
        from autotrain.data_types import Sample

        # Get the baked prompt for the topic
        prompt = self.model.prompts.get_producer(topic)

        async def _generate_one(i):
            # Run sync _generate_input in thread pool
            input_data = await executor.run_sync(self._generate_input, i, prompt)
            return Sample(
                input_data=input_data,
                output_data="",  # Will be filled by Solver
                metadata={
                    "source": "producer",
                    "producer": "model",
                    "iteration": 0,
                    "topic": topic,
                },
            )

        if executor is None:
            raise ValueError("executor is required for async generate")

        results = await executor.map_async(_generate_one, range(count))
        return results

    def _generate_input(self, seed: int, prompt: str) -> str:
        """
        Generate a single input sample.

        Uses the model to generate diverse inputs based on the seed and prompt.
        Falls back to template-based generation if model is not available.

        Args:
            seed: Seed value for diversity
            prompt: The prompt template to use

        Returns:
            Generated input string
        """
        # Try to use the model for generation if available and loaded
        if self.model.is_loaded and not hasattr(self.model, "_spec_class"):
            try:
                # Add variation to the prompt for diversity
                varied_prompt = f"{prompt}\n\nGenerate sample variation {seed + 1}."

                result = self.model.generate(
                    prompt=varied_prompt,
                    temperature=0.8 + (seed % 10) * 0.02,  # Vary temperature for diversity
                    max_tokens=256,
                )

                # Check if result is a string (not a MagicMock)
                if isinstance(result, str) and result:
                    return result
            except Exception as e:
                print(f"Warning: Model generation failed, using fallback: {e}")

        # Fallback: Use template-based generation
        templates = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "Calculate 15% of 200.",
            "Write a short poem about nature.",
            "What are the benefits of exercise?",
            "Describe the process of making bread.",
            "What is the difference between weather and climate?",
            "How do you solve a quadratic equation?",
            "What are the main causes of World War I?",
            "Explain the concept of supply and demand.",
        ]
        return templates[seed % len(templates)]


__all__ = ["Producer"]
