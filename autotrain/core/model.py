"""Core Model class for AutoTrain."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from autotrain.checkpoints import CheckpointManager
from autotrain.components import Checker, Producer, Solver, Splitter
from autotrain.config import (
    InferenceConfig,
    PEFTConfig,
    Prompts,
    ScalableTrainingConfig,
    TrainingConfig,
)
from autotrain.data_types import Sample
from autotrain.expert import Expert
from autotrain.tools import Tool, ToolCallingConfig

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.templates import InstructionTemplate


class Model:
    """
    Model class wrapping an unsloth fine-tunable model.

    Implements self-tuning through iteration-by-iteration loop:
    Producer -> Solver -> Splitter -> Checker -> Fine-tune
    """

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3.5-27B-GGUF",
        sample_multiplier: int = 2,
        inference_config: Optional[InferenceConfig] = None,
        prompts: Optional[Prompts] = None,
        enable_checker: bool = False,
        checker_rewrite_mode: bool = False,
        checkpoint_dir: Optional[str] = None,
        keep_best_checkpoint: bool = True,
        scalable_config: Optional[ScalableTrainingConfig] = None,
        thinking: bool = True,
    ):
        """
        Initialize the Model.

        Args:
            model_name: Name/path of the unsloth model to load (default: unsloth/Qwen3.5-27B-GGUF)
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
        self.sample_multiplier = sample_multiplier  # Number of samples to generate per input
        self.inference_config = inference_config or InferenceConfig()
        self.prompts = prompts or Prompts()
        self.enable_checker = enable_checker
        self.checker_rewrite_mode = checker_rewrite_mode
        self._scalable_config = scalable_config or ScalableTrainingConfig()
        self.thinking = thinking

        # Checkpoint manager
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir or "./checkpoints", keep_best=keep_best_checkpoint
        )

        # Benchmark
        self._benchmark: Optional["Benchmark"] = None

        # PEFT and Training configuration
        self._peft_config = PEFTConfig()
        self._training_config = TrainingConfig()
        self._current_iteration = 0

        # Template for instruction formatting
        self._template: Optional["InstructionTemplate"] = None
        self._init_template()

        # Underlying unsloth model (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._fast_model = None
        self._is_model_loaded = False

        # Store original model name for reloading from scratch each iteration
        self._base_model_name = model_name

        # Components
        self._producer: Optional[Producer] = None
        self._solver: Optional[Solver] = None
        self._splitter: Optional[Splitter] = None
        self._checker: Optional[Checker] = None

        # Dataset storage
        self._samples: list[Sample] = []
        self._training_data: list[dict] = []
        self.expert_training_data: list[dict] = []
        self.collect_expert_data = False

        # Training history
        self._benchmark_history: list[dict] = []

        # Tool calling
        self._tools: ToolCallingConfig = ToolCallingConfig()

    # ==================== Template Initialization ====================

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

    # ==================== Model Loading ====================

    @property
    def model(self):
        """Access the underlying unsloth model."""
        return self._model

    @property
    def tokenizer(self):
        """Access the underlying tokenizer."""
        return self._tokenizer

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
        """Load the unsloth model."""
        try:
            from unsloth import FastLanguageModel

            self._fast_model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self._base_model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            self._model = self._fast_model.model  # type: ignore
            self._is_model_loaded = True
            print(f"Model loaded: {self._base_model_name}")
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

    # ==================== PEFT Configuration ====================

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

    # ==================== Scalable Training Configuration ====================

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
        print(
            f"Scalable config updated: gradient_checkpointing={gradient_checkpointing}, "
            f"mixed_precision={mixed_precision}"
        )

    def get_scalable_config(self) -> ScalableTrainingConfig:
        """Get current scalable training configuration."""
        return self._scalable_config

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

    # ==================== Delegated Methods ====================
    # These methods delegate to functions in core.management.py

    def add_expert(
        self,
        expert: "Expert",
        production_weight: float = 1.0,
        check_weight: float = 0.0,
    ) -> None:
        """Add an expert with weighted production rates."""
        from autotrain.core.management import add_expert

        add_expert(self, expert, production_weight, check_weight)

    def remove_expert(self, expert: "Expert") -> None:
        """Remove an expert from all components."""
        from autotrain.core.management import remove_expert

        remove_expert(self, expert)

    def clear_experts(self) -> None:
        """Clear all experts."""
        from autotrain.core.management import clear_experts

        clear_experts(self)

    def set_expert_weights(self, weights: dict["Expert", float]) -> None:
        """Set expert weights."""
        from autotrain.core.management import set_expert_weights

        set_expert_weights(self, weights)

    def add_experts(
        self,
        experts: list[tuple["Expert", float]],
        check_weight: float = 0.0,
    ) -> None:
        """Add multiple experts with production weights."""
        from autotrain.core.management import add_experts

        add_experts(self, experts, check_weight)

    def set_benchmark(self, benchmark: "Benchmark") -> None:
        """Set the benchmark for evaluation."""
        from autotrain.core.management import set_benchmark

        set_benchmark(self, benchmark)

    def get_benchmark(self) -> Optional["Benchmark"]:
        """Get the current benchmark."""
        from autotrain.core.management import get_benchmark

        return get_benchmark(self)

    def add_benchmark_sample(
        self,
        input_data: str,
        expected_output: str,
        mode: str = "exact_match",
    ) -> None:
        """Add a sample to the benchmark."""
        from autotrain.core.management import add_benchmark_sample

        add_benchmark_sample(self, input_data, expected_output, mode)

    def load_benchmark(self, path: Union[str, Path]) -> None:
        """Load benchmark from file."""
        from autotrain.core.management import load_benchmark

        load_benchmark(self, str(path))

    def save_checkpoint(
        self, iteration: int, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Save a checkpoint."""
        from autotrain.core.management import save_checkpoint

        save_checkpoint(self, iteration, metadata)
        return f"checkpoint_{iteration}" if metadata else None

    def load_checkpoint(
        self, checkpoint_id: Optional[str] = None, iteration: Optional[int] = None
    ) -> Any:
        """Load a checkpoint."""
        from autotrain.core.management import load_checkpoint

        return load_checkpoint(self, checkpoint_id, iteration)

    def restore_best_checkpoint(self) -> None:
        """Restore the best checkpoint."""
        from autotrain.core.management import restore_best_checkpoint

        restore_best_checkpoint(self)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        from autotrain.core.management import list_checkpoints

        return list_checkpoints(self)

    def _call_model(self, prompt: str, inference_config: Optional["InferenceConfig"] = None) -> str:
        """Generate output from the model."""
        from autotrain.core.management import _call_model

        return _call_model(self, prompt, inference_config)

    def _init_components(
        self,
        experts: Optional[list[tuple["Expert", float]]] = None,
    ) -> None:
        """Initialize pipeline components (producer, solver, splitter, checker)."""
        from autotrain.core.training import _init_components

        _init_components(self, experts)

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text from the model."""
        from autotrain.core.management import generate

        return generate(self, prompt, temperature, max_tokens, top_p)

    def export_expert_data(self, path: Union[str, Path]) -> None:
        """
        Export collected expert training data to a file.

        Args:
            path: Output file path (.json or .jsonl)
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if str(path).endswith(".jsonl"):
            with open(path, "w") as f:
                for item in self.expert_training_data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(path, "w") as f:
                json.dump(self.expert_training_data, f, indent=2)

        print(f"Expert training data exported to {path} ({len(self.expert_training_data)} items)")

    # ==================== Tool Management ====================

    def add_tool(
        self,
        tool: Union[Tool, Callable[..., Any]],
        name: Optional[str] = None,
    ) -> Tool:
        """
        Add a tool for tool calling.

        Args:
            tool: A Tool instance or a callable function
            name: Optional name override

        Returns:
            The added Tool

        Example:
            from autotrain.tools import python

            model.add_tool(python)
        """
        return self._tools.add_tool(tool, name)

    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool by name.

        Args:
            name: Tool name to remove

        Returns:
            True if removed, False if not found
        """
        return self._tools.remove_tool(name)

    def clear_tools(self) -> None:
        """Remove all tools."""
        self._tools.clear_tools()

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return self._tools.list_tools()

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return self._tools.has_tool(name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get_tool(name)

    def create_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Tool]:
        """
        Decorator to create a tool from a function.

        Args:
            name: Optional name override
            description: Optional description override

        Returns:
            Decorator function

        Example:
            @model.create_tool(description="Get current time")
            def get_time(timezone: str = "UTC") -> str:
                import datetime
                return datetime.datetime.now(timezone).isoformat()
        """
        from autotrain.tools import create_tool as _create_tool

        return _create_tool(name=name, description=description)

    def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tool_calls: int = 10,
        tool_choice: Optional[str] = None,
    ) -> str:
        """
        Generate text with tool calling support.

        Args:
            prompt: User prompt
            tools: Optional list of tool schemas (if None, uses registered tools)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling
            max_tool_calls: Maximum number of tool calls (default: 10)
            tool_choice: Force specific tool ("none" to disable tools)

        Returns:
            Generated text with tool execution results
        """
        from autotrain.core.management import generate_with_tools as _generate_with_tools

        return _generate_with_tools(
            self,
            prompt,
            tools,
            temperature,
            max_tokens,
            top_p,
            max_tool_calls,
            tool_choice,
        )

    def export_gguf(
        self,
        output_path: Union[str, Path],
        quantization: str = "q4_k_m",
        merge_adapter: bool = True,
    ) -> str:
        """Export model to GGUF format."""
        from autotrain.core.management import export_gguf

        return export_gguf(self, str(output_path), quantization, merge_adapter)

    def push_to_huggingface(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        merge_adapter: bool = True,
    ) -> None:
        """Push model to Hugging Face Hub."""
        from autotrain.core.management import push_to_huggingface

        push_to_huggingface(self, repo_id, token, private, merge_adapter)

    def export_to_ollama(
        self, output_path: Union[str, Path], model_name: str, quantization: str = "q4_k_m"
    ) -> None:
        """Export model to Ollama format."""
        from autotrain.core.management import export_to_ollama

        export_to_ollama(self, str(output_path), model_name, quantization)

    def train(self, **kwargs: Any) -> Any:
        """Train the model."""
        from autotrain.core.training import train

        return train(self, **kwargs)


__all__ = ["Model"]
