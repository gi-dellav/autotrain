"""Core Model class for AutoTrain."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from autotrain.core.base_model import BaseModel
from autotrain.data_types import Sample
from autotrain.expert import Expert
from autotrain.tools import Tool

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.templates import InstructionTemplate


class Model(BaseModel):
    """
    Model class wrapping an unsloth fine-tunable model.

    Implements self-tuning through iteration-by-iteration loop:
    Producer -> Solver -> Splitter -> Checker -> Fine-tune
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3.5-27B-GGUF",
        sample_multiplier: int = 2,
        inference_config: Optional[Any] = None,
        prompts: Optional[Any] = None,
        enable_checker: bool = False,
        checker_rewrite_mode: bool = False,
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
            checker_rewrite_mode=checker_rewrite_mode,
            checkpoint_dir=checkpoint_dir,
            keep_best_checkpoint=keep_best_checkpoint,
            scalable_config=scalable_config,
            thinking=thinking,
        )
        self._init_template()

    def _init_template(self) -> None:
        """Initialize template based on model name."""
        from autotrain.templates import auto_detect_template

        self._template = auto_detect_template(self.model_name)
        print(f"Using template: {self._template.name} (auto-detected from {self.model_name})")

    def set_template(
        self,
        template: Optional[Union["InstructionTemplate", str]] = None,
        model_name: Optional[str] = None,
    ) -> "InstructionTemplate":
        """Set or update the instruction template."""
        from autotrain.templates import auto_detect_template, get_template

        if template is None:
            detect_name = model_name or self.model_name
            self._template = auto_detect_template(detect_name)
            print(f"Template auto-detected: {self._template.name}")
        elif isinstance(template, str):
            self._template = get_template(template)
            print(f"Template set by name: {template}")
        elif hasattr(template, "name"):  # InstructionTemplate instance
            self._template = template
            print(f"Template set: {template.name}")
        else:
            raise ValueError(
                f"template must be str, InstructionTemplate, or None, got {type(template).__name__}"
            )

        return self._template

    def get_template(self) -> Optional["InstructionTemplate"]:
        """Get the current instruction template."""
        return self._template

    def load_model(  # type: ignore[override]
        self,
        max_seq_length: int = 2048,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = False,
    ) -> None:
        """Load the unsloth model.
        
        Args:
            max_seq_length: Maximum sequence length.
            dtype: Data type for the model.
            load_in_4bit: If True, uses qLoRA (4-bit quantization). If False, uses standard LoRA.
                         Defaults to False (LoRA).
        """
        try:
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]

            self._fast_model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self._base_model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            self._model = self._fast_model.model  # type: ignore
            self._is_model_loaded = True
            print(f"Model loaded: {self._base_model_name} ({'qLoRA' if load_in_4bit else 'LoRA'})")
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

        self._is_model_loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("Model unloaded from memory")

    def auto_tune_batch_size(self) -> int:
        """Automatically find the largest batch size that fits in GPU memory."""
        if not self._is_model_loaded:
            print("Model not loaded, using default batch size")
            return self._training_config.batch_size

        try:
            import torch
        except ImportError:
            print("torch not available, using default batch size")
            return self._training_config.batch_size

        max_memory_mb = self._scalable_config.max_memory_mb
        if max_memory_mb is None:
            try:
                if torch.cuda.is_available():
                    max_memory_mb = int(
                        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 * 0.8
                    )
                else:
                    max_memory_mb = 8192
            except Exception:
                max_memory_mb = 8192

        batch_size = 1
        best_batch_size = 1
        while batch_size <= 32:
            try:
                estimated_memory = self._estimate_memory_for_batch(batch_size)
                if estimated_memory > max_memory_mb:
                    break
                best_batch_size = batch_size
                batch_size *= 2
            except Exception:
                break

        self._training_config.batch_size = best_batch_size
        print(f"Auto-tuned batch size: {best_batch_size}")
        return best_batch_size

    def _estimate_memory_for_batch(self, batch_size: int) -> int:
        """Estimate GPU memory required for a batch size."""
        base_memory = 4000
        per_sample_memory: float = 100
        if self._scalable_config.mixed_precision in ("fp16", "bf16"):
            per_sample_memory *= 0.5
        elif self._scalable_config.mixed_precision == "fp32":
            per_sample_memory *= 2
        return int(base_memory + (batch_size * per_sample_memory))

    # Delegated methods
    def add_expert(
        self,
        expert: "Expert",
        weight: float = 1.0,
        production_weight: Optional[float] = None,
        check_weight: float = 0.0,
    ) -> None:
        from autotrain.core.management import add_expert

        add_expert(
            self, expert, production_weight=production_weight or weight, check_weight=check_weight
        )

    def generate(  # type: ignore[override]
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        iteration: Optional[int] = None,
    ) -> str:
        from autotrain.core.management import generate
        from autotrain.utils.function_evaluator import evaluate_temperature

        eff_temperature: Optional[float] = None
        if temperature is not None:
            eff_temperature = temperature
        elif self.inference_config.temperature_fn is not None:
            if iteration is None:
                iteration = self._current_iteration
            eff_temperature = evaluate_temperature(self.inference_config.temperature_fn, iteration)

        return generate(self, prompt, eff_temperature, max_tokens, top_p)

    def train(self, **kwargs: Any) -> Any:
        from autotrain.core.training import train

        return train(self, **kwargs)

    def export_gguf(
        self,
        output_path: Union[str, Path],
        quantization: str = "q4_k_m",
        merge_adapter: bool = True,
    ) -> str:
        from autotrain.core.management import export_gguf

        return export_gguf(self, str(output_path), quantization, merge_adapter)

    def push_to_huggingface(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        merge_adapter: bool = True,
    ) -> None:
        from autotrain.core.management import push_to_huggingface

        push_to_huggingface(self, repo_id, token, private, merge_adapter)

    def export_to_ollama(
        self, output_path: Union[str, Path], model_name: str, quantization: str = "q4_k_m"
    ) -> None:
        from autotrain.core.management import export_to_ollama

        export_to_ollama(self, str(output_path), model_name, quantization)


__all__ = ["Model"]
