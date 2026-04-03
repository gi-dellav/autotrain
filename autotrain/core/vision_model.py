"""Vision Model class for VLM fine-tuning."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from autotrain.core.base_model import BaseModel
from autotrain.data_types import Sample, VisionSample
from autotrain.expert import Expert, VisionExpert

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.templates import InstructionTemplate


class VisionModel(BaseModel):
    """
    Vision Model class wrapping an unsloth fine-tunable VLM.

    Implements self-tuning through iteration-by-iteration loop for VLMs:
    Producer -> Solver -> Splitter -> Checker -> Fine-tune
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3.5-27B-GGUF",
        sample_multiplier: int = 2,
        inference_config: Optional[Any] = None,
        prompts: Optional[Any] = None,
        enable_checker: bool = False,
        checkpoint_dir: Optional[str] = None,
        keep_best_checkpoint: bool = True,
        scalable_config: Optional[Any] = None,
        thinking: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            sample_multiplier=sample_multiplier,
            inference_config=inference_config,
            prompts=prompts,
            enable_checker=enable_checker,
            checkpoint_dir=checkpoint_dir,
            keep_best_checkpoint=keep_best_checkpoint,
            scalable_config=scalable_config,
            thinking=thinking,
        )
        self._processor = None

    @property
    def processor(self):
        """Access the processor."""
        return self._processor

    def load_model(
        self,
        max_seq_length: int = 2048,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = True,
    ) -> None:
        """Load the unsloth VLM model."""
        try:
            from unsloth import FastVisionModel

            self._fast_model, self._tokenizer = FastVisionModel.from_pretrained(
                model_name=self._base_model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            self._model = self._fast_model.model
            self._processor = self._tokenizer
            self._is_model_loaded = True
            print(f"Vision model loaded: {self._base_model_name}")
        except ImportError:
            raise ImportError("unsloth is required. Install with: pip install unsloth")

    def _unload_model(self) -> None:
        """Unload the model from memory to free resources."""
        import gc

        if self._fast_model is not None:
            del self._fast_model
            self._fast_model = None
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        self._is_model_loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("Vision model unloaded from memory")

    def add_sample(self, sample: Union[Sample, VisionSample]) -> None:
        """Add a training sample."""
        if isinstance(sample, VisionSample):
            self._samples.append(sample)
            self._training_data.append(
                {
                    "input_data": sample.input_data,
                    "output_data": sample.output_data,
                    "images": sample.images,
                    "messages": sample.to_conversation(),
                    "metadata": sample.metadata,
                }
            )
        else:
            vision_sample = VisionSample(
                input_data=sample.input_data,
                output_data=sample.output_data,
                metadata=sample.metadata,
            )
            self._samples.append(vision_sample)
            self._training_data.append(
                {
                    "input_data": sample.input_data,
                    "output_data": sample.output_data,
                    "messages": vision_sample.to_conversation(),
                    "metadata": sample.metadata,
                }
            )

    def generate(
        self,
        prompt: str,
        images: Optional[List] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Generate text using the model."""
        if not self._is_model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        messages = [{"role": "user", "content": [{"type": "image"}] if images else []}]
        if images:
            messages[0]["content"].append({"type": "text", "text": prompt})
        else:
            messages[0]["content"] = [{"type": "text", "text": prompt}]

        input_text = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._tokenizer(
            images[0] if images else None,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda" if self._is_model_loaded else "cpu")

        if self._fast_model is None:
            raise RuntimeError("Model not loaded")

        outputs = self._fast_model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=max_tokens,
            use_cache=True,
            thinking=self.thinking if self.inference_config.thinking else False,
            **kwargs,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train(self, **kwargs):
        from autotrain.core.training import train_vision_model

        return train_vision_model(self, **kwargs)

    def import_dataset(self, path: Union[str, Path]) -> None:
        import json

        path = Path(path)
        data = json.loads(path.read_text())
        for item in data:
            images = item.get("images", [])
            sample = VisionSample(
                input_data=item.get("input_data", ""),
                output_data=item.get("output_data", ""),
                images=images,
                metadata=item.get("metadata", {}),
            )
            self.add_sample(sample)
        print(f"Imported {len(data)} samples from {path}")


__all__ = ["VisionModel"]
