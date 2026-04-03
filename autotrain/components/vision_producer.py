"""Vision Producer component - generates vision-language samples."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from autotrain.config import InferenceConfig
    from autotrain.core.vision_model import VisionModel
    from autotrain.data_types import VisionSample


class VisionProducer:
    """
    Vision Producer component - generates vision-language input samples.

    Creates diverse input samples with images for the model to solve.
    """

    DEFAULT_VISION_TASKS = [
        "Describe the main objects in this image",
        "What is happening in this image?",
        "Count the number of objects in this image",
        "What colors are present in this image?",
        "Describe the background of this image",
        "What is the subject of this image?",
        "Describe the setting of this image",
        "What details stand out in this image?",
    ]

    def __init__(
        self,
        model: "VisionModel",
        prompt: str,
        inference_config: "InferenceConfig",
    ):
        self.model = model
        self.prompt = prompt
        self.inference_config = inference_config

    def generate(
        self,
        count: int,
        images: Optional[List[Any]] = None,
        task_description: Optional[str] = None,
    ) -> List["VisionSample"]:
        """
        Generate vision-language samples in parallel.

        Args:
            count: Number of samples to generate
            images: Optional list of images to use
            task_description: Optional task description for generation

        Returns:
            List of generated VisionSample
        """
        from concurrent.futures import ThreadPoolExecutor

        from autotrain.data_types import VisionSample

        task_desc = task_description or self.prompt or "Describe this image"

        def _generate_one(i):
            image = images[i] if images and i < len(images) else None
            input_data = self._generate_input(i, task_desc)
            return VisionSample(
                input_data=input_data,
                output_data="",
                images=[image] if image else [],
                metadata={"source": "producer", "producer": "vision_model", "iteration": 0},
            )

        max_workers = self.model.inference_config.max_workers
        if not isinstance(max_workers, int):
            max_workers = 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            samples = list(executor.map(_generate_one, range(count)))

        return samples

    def _generate_input(self, seed: int, task_description: str) -> str:
        """
        Generate a single input sample.

        Uses the model to generate diverse inputs based on the seed.
        Falls back to template-based generation if model is not available.

        Args:
            seed: Seed value for diversity
            task_description: Task description to use

        Returns:
            Generated input string
        """
        if self.model.is_loaded:
            try:
                diverse_tasks = self.DEFAULT_VISION_TASKS
                task = diverse_tasks[seed % len(diverse_tasks)]

                result = self.model.generate(
                    prompt=task,
                    temperature=0.8 + (seed % 10) * 0.02,
                    max_tokens=256,
                )

                if isinstance(result, str) and result:
                    return result
            except Exception as e:
                print(f"Warning: Model generation failed, using fallback: {e}")

        return self._get_fallback_task(seed)

    def _get_fallback_task(self, seed: int) -> str:
        """Get a fallback task for no-data mode."""
        return self.DEFAULT_VISION_TASKS[seed % len(self.DEFAULT_VISION_TASKS)]

    def generate_with_prompts(
        self,
        count: int,
        custom_prompts: List[str],
        images: Optional[List[Any]] = None,
    ) -> List["VisionSample"]:
        """
        Generate vision samples with custom prompts.

        Args:
            count: Number of samples to generate
            custom_prompts: List of custom prompt strings
            images: Optional list of images

        Returns:
            List of generated VisionSample
        """
        from autotrain.data_types import VisionSample

        samples = []

        for i in range(count):
            prompt = custom_prompts[i % len(custom_prompts)]

            sample_images = []
            if images:
                sample_images = [images[i % len(images)]]

            sample = VisionSample(
                input_data=prompt,
                output_data="",
                images=sample_images,
                metadata={"source": "producer", "producer": "custom_prompt", "index": i},
            )
            samples.append(sample)

        return samples


__all__ = ["VisionProducer"]
