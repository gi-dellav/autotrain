"""Vision Model class for VLM fine-tuning."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from autotrain.checkpoints import CheckpointManager
from autotrain.components import Checker, Producer, Solver, Splitter
from autotrain.config import (
    InferenceConfig,
    PEFTConfig,
    Prompts,
    ScalableTrainingConfig,
    TrainingConfig,
)
from autotrain.data_types import Sample, VisionSample
from autotrain.expert import Expert, VisionExpert

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.templates import InstructionTemplate


class VisionModel:
    """
    Vision Model class wrapping an unsloth fine-tunable VLM.

    Implements self-tuning through iteration-by-iteration loop for VLMs:
    Producer -> Solver -> Splitter -> Checker -> Fine-tune

    Example usage:
        model = VisionModel(model_name="unsloth/Qwen3.5-27B-GGUF")
        model.train(k=10, i=5, experts=[expert])
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3.5-27B-GGUF",
        sample_multiplier: int = 2,
        inference_config: Optional[InferenceConfig] = None,
        prompts: Optional[Prompts] = None,
        enable_checker: bool = False,
        checkpoint_dir: Optional[str] = None,
        keep_best_checkpoint: bool = True,
        scalable_config: Optional[ScalableTrainingConfig] = None,
        thinking: bool = True,
    ):
        """
        Initialize the Vision Model.

        Args:
            model_name: Name/path of the unsloth VLM to load (default: unsloth/Qwen3.5-27B-GGUF)
            sample_multiplier: Number of samples to generate per input (default: 2)
            inference_config: Configuration for inference properties
            prompts: Custom prompts for each component
            enable_checker: Whether to enable the Checker component
            checkpoint_dir: Directory for checkpoints
            keep_best_checkpoint: Always keep the best benchmark checkpoint
            scalable_config: Configuration for scalable training features
            thinking: Whether to enable thinking/reasoning (default: True)
        """
        self.model_name = model_name
        self.sample_multiplier = sample_multiplier
        self.inference_config = inference_config or InferenceConfig()
        self.prompts = prompts or Prompts()
        self.enable_checker = enable_checker
        self._scalable_config = scalable_config or ScalableTrainingConfig()
        self.thinking = thinking

        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir or "./checkpoints", keep_best=keep_best_checkpoint
        )

        self._benchmark: Optional["Benchmark"] = None

        self._peft_config = PEFTConfig()
        self._training_config = TrainingConfig()
        self._current_iteration = 0

        self._template: Optional["InstructionTemplate"] = None

        self._model = None
        self._tokenizer = None
        self._fast_model = None
        self._processor = None
        self._is_model_loaded = False

        self._base_model_name = model_name

        self._producer: Optional[Producer] = None
        self._solver: Optional[Solver] = None
        self._splitter: Optional[Splitter] = None
        self._checker: Optional[Checker] = None

        self._samples: list[VisionSample] = []
        self._training_data: list[dict] = []

        self._benchmark_history: list[dict] = []

    @property
    def model(self):
        """Access the underlying unsloth model."""
        return self._model

    @property
    def tokenizer(self):
        """Access the underlying tokenizer."""
        return self._tokenizer

    @property
    def processor(self):
        """Access the processor."""
        return self._processor

    @property
    def fast_model(self):
        """Access the fast model wrapper."""
        return self._fast_model

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_model_loaded

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

    def set_peft_config(self, config: PEFTConfig) -> None:
        """Configure PEFT/LoRA parameters."""
        self._peft_config = config
        print(f"PEFT config updated: r={config.r}, alpha={config.lora_alpha}")

    def get_peft_config(self) -> PEFTConfig:
        """Get current PEFT configuration."""
        return self._peft_config

    def set_training_config(self, config: TrainingConfig) -> None:
        """Configure training hyperparameters."""
        self._training_config = config
        print(f"Training config updated: lr={config.learning_rate}, epochs={config.epochs}")

    def get_training_config(self) -> TrainingConfig:
        """Get current training configuration."""
        return self._training_config

    def set_scalable_config(
        self,
        gradient_checkpointing: bool = True,
        mixed_precision: str = "fp16",
        batch_size_auto_tune: bool = False,
        max_memory_mb: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_flash_attention: bool = False,
    ) -> None:
        """Configure scalable training features."""
        self._scalable_config = ScalableTrainingConfig(
            gradient_checkpointing=gradient_checkpointing,
            mixed_precision=mixed_precision,
            batch_size_auto_tune=batch_size_auto_tune,
            max_memory_mb=max_memory_mb,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_flash_attention=use_flash_attention,
        )
        print(f"Scalable config updated: gc={gradient_checkpointing}, mp={mixed_precision}")

    def get_scalable_config(self) -> ScalableTrainingConfig:
        """Get current scalable configuration."""
        return self._scalable_config

    def add_expert(self, expert: "VisionExpert", weight: float = 1.0) -> None:
        """Add an expert to the model."""
        if not hasattr(self, "_experts"):
            self._experts: List[tuple] = []
        self._experts.append((expert, weight))
        print(f"Added expert: {expert.model_name} (weight: {weight})")

    def add_experts(self, experts: List[tuple]) -> None:
        """Add multiple experts to the model."""
        for expert, weight in experts:
            self.add_expert(expert, weight)

    def remove_expert(self, expert: "VisionExpert") -> None:
        """Remove an expert from the model."""
        if hasattr(self, "_experts"):
            self._experts = [(e, w) for e, w in self._experts if e != expert]
            print(f"Removed expert: {expert.model_name}")

    def clear_experts(self) -> None:
        """Remove all experts from the model."""
        if hasattr(self, "_experts"):
            self._experts = []
            print("Cleared all experts")

    def get_experts(self) -> List[tuple]:
        """Get all experts with their weights."""
        return getattr(self, "_experts", [])

    def set_expert_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for experts by model name."""
        if hasattr(self, "_experts"):
            for i, (expert, _) in enumerate(self._experts):
                if expert.model_name in weights:
                    self._experts[i] = (expert, weights[expert.model_name])
            print(f"Updated expert weights: {weights}")

    def set_benchmark(self, benchmark: "Benchmark") -> None:
        """Set evaluation benchmark."""
        self._benchmark = benchmark
        print(f"Benchmark set: {benchmark.__class__.__name__}")

    def get_benchmark(self) -> Optional["Benchmark"]:
        """Get current benchmark."""
        return self._benchmark

    def add_sample(self, sample: Union[Sample, VisionSample]) -> None:
        """Add a training sample."""
        if isinstance(sample, VisionSample):
            self._samples.append(sample)
        else:
            vision_sample = VisionSample(
                input_data=sample.input_data,
                output_data=sample.output_data,
                metadata=sample.metadata,
            )
            self._samples.append(vision_sample)

    def get_samples(self) -> List[VisionSample]:
        """Get all training samples."""
        return self._samples

    def clear_samples(self) -> None:
        """Clear all training samples."""
        self._samples = []
        print("Cleared all samples")

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

        from transformers import TextStreamer

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

    def get_peft_model(self, **kwargs):
        """Get PEFT model for vision training."""
        if self._fast_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from unsloth import FastVisionModel

        return FastVisionModel.get_peft_model(
            self._fast_model,
            **kwargs,
        )

    def for_inference(self) -> None:
        """Prepare model for inference."""
        if self._fast_model is not None:
            try:
                from unsloth import FastVisionModel

                FastVisionModel.for_inference(self._fast_model)
                print("Model prepared for inference")
            except Exception as e:
                print(f"Warning: Could not prepare for inference: {e}")

    def train(
        self,
        k: int = 10,
        i: int = 5,
        experts: Optional[List[tuple]] = None,
        initial_samples: Optional[List[VisionSample]] = None,
        benchmark: Optional["Benchmark"] = None,
        early_stopping: bool = True,
        checkpoint_every: int = 1,
        resume_from_checkpoint: bool = False,
    ):
        """
        Train the model using the self-tuning loop.

        Args:
            k: Number of samples to use for fine-tuning per iteration
            i: Number of iterations
            experts: Optional list of (expert, weight) tuples
            initial_samples: Optional initial dataset to start with
            benchmark: Optional evaluation benchmark
            early_stopping: Stop if no improvement
            checkpoint_every: Save checkpoint every N iterations
            resume_from_checkpoint: Resume from latest checkpoint
        """
        from autotrain.core.training import train_vision_model

        return train_vision_model(
            model=self,
            k=k,
            i=i,
            experts=experts,
            initial_samples=initial_samples,
            benchmark=benchmark or self._benchmark,
            early_stopping=early_stopping,
            checkpoint_every=checkpoint_every,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def export_dataset(self, path: Union[str, Path]) -> None:
        """Export training data to a file."""
        import json

        path = Path(path)
        data = [sample.to_dict() for sample in self._training_data]
        path.write_text(json.dumps(data, indent=2, default=str))
        print(f"Dataset exported to {path} ({len(data)} samples)")

    def import_dataset(self, path: Union[str, Path]) -> None:
        """Import training data from a file."""
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

    def get_history(self) -> List[dict]:
        """Get training history."""
        return self._benchmark_history

    def __repr__(self) -> str:
        return f"VisionModel(model_name='{self.model_name}', loaded={self._is_model_loaded})"


__all__ = ["VisionModel"]
