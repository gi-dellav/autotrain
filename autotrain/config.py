"""Configuration classes for AutoTrain."""

import random
from dataclasses import dataclass
from typing import Dict, Optional, Union


def _get_prompt(prompt: Optional[Union[str, list[str]]]) -> Optional[str]:
    """Get a single prompt from str or list[str], selecting randomly if list."""
    if prompt is None:
        return None
    if isinstance(prompt, list):
        if not prompt:
            return None
        return random.choice(prompt)
    return prompt


@dataclass
class InferenceConfig:
    """Configuration for inference properties."""

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    thinking: bool = True


@dataclass
class Prompts:
    """Configurable prompts for each component.

    For Producer and Solver, use the baking functions by calling get_producer() or get_solver().
    For Splitter and Checker, use get_splitter() or get_checker().

    If custom prompts are set, they will be used directly. Otherwise, the default detailed
    prompts from the prompts module will be used.

    Each prompt can be either a single string or a list of strings. If a list is provided,
    a random prompt will be selected each time get_* is called.

    Example:
        # Use default baked prompts
        prompts = Prompts()
        producer_prompt = prompts.get_producer("Python programming")

        # Use custom prompts
        custom_prompts = Prompts(
            producer="Custom producer prompt",
            solver="Custom solver prompt",
        )

        # Use multiple prompts (randomly selected)
        multi_prompts = Prompts(
            producer=["Generate a Python question", "Create a coding challenge"],
            solver=["Solve this problem", "Answer the following"],
        )
    """

    producer: Optional[Union[str, list[str]]] = None
    solver: Optional[Union[str, list[str]]] = None
    splitter: Optional[Union[str, list[str]]] = None
    checker: Optional[Union[str, list[str]]] = None

    def get_producer(self, topic: str, format: str = "json") -> str:
        """Get the producer prompt, using bake_producer if no custom prompt is set.

        Args:
            topic: The topic to generate training samples for.
            format: The desired output format (json, text, etc.).

        Returns:
            The producer prompt string.
        """
        from .prompts import bake_producer

        return _get_prompt(self.producer) if self.producer else bake_producer(topic, format)

    def get_solver(self, topic: Optional[str] = None, context: str = "") -> str:
        """Get the solver prompt, using bake_solver if no custom prompt is set.

        Args:
            topic: Optional topic context for the solver.
            context: Additional context or instructions.

        Returns:
            The solver prompt string.
        """
        from .prompts import bake_solver

        return _get_prompt(self.solver) if self.solver else bake_solver(topic, context)

    def get_splitter(self) -> str:
        """Get the splitter prompt using the default constant if no custom prompt is set.

        Returns:
            The splitter prompt string.
        """
        from .prompts import SPLITTER_DEFAULT

        return _get_prompt(self.splitter) if self.splitter else SPLITTER_DEFAULT

    def get_checker(self) -> str:
        """Get the checker prompt using the default constant if no custom prompt is set.

        Returns:
            The checker prompt string.
        """
        from .prompts import CHECKER_DEFAULT

        return _get_prompt(self.checker) if self.checker else CHECKER_DEFAULT


@dataclass
class PEFTConfig:
    """Configuration for PEFT/LoRA fine-tuning.

    Default values follow Unsloth's recommended settings.
    See: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
    """

    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    target_modules: Optional[list[str]] = None
    use_rslora: bool = False
    loftq_config: Optional[dict] = None

    def get_target_modules(self, model_type: str = "llama") -> list[str]:
        """Get target modules based on model type."""
        if self.target_modules:
            return self.target_modules

        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.

    Default values follow Unsloth's recommended settings.
    See: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
    """

    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.15
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_total_limit: int = 3
    seed: int = 3407
    scheduler_type: str = "cosine"


@dataclass
class ScalableTrainingConfig:
    """
    Configuration for scalable training features.

    Provides memory-efficient training options for larger models.

    Default values follow Unsloth's recommended settings.
    See: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

    Args:
        gradient_checkpointing: Enable gradient checkpointing to save memory ("unsloth", True, or False)
        mixed_precision: Mixed precision mode ("fp16", "bf16", "fp32"). bf16 recommended for modern GPUs.
        batch_size_auto_tune: Automatically find optimal batch size
        max_memory_mb: Maximum GPU memory to use (MB) for auto-tuning
        num_workers: Number of DataLoader workers for parallel loading
        dataloader_num_workers: Alias for num_workers
        pin_memory: Pin memory for faster CPU->GPU transfer
        use_flash_attention: Enable flash attention if available
    """

    gradient_checkpointing: str = "unsloth"
    mixed_precision: str = "bf16"
    batch_size_auto_tune: bool = False
    max_memory_mb: Optional[int] = None
    num_workers: int = 4
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    use_flash_attention: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.mixed_precision not in ["fp16", "bf16", "fp32"]:
            raise ValueError(
                f"mixed_precision must be 'fp16', 'bf16', or 'fp32', got '{self.mixed_precision}'"
            )
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.gradient_checkpointing not in [True, False, "unsloth"]:
            raise ValueError(
                f"gradient_checkpointing must be True, False, or 'unsloth', "
                f"got '{self.gradient_checkpointing}'"
            )

        # Sync dataloader_num_workers with num_workers if not explicitly set
        if self.dataloader_num_workers == 4 and self.num_workers != 4:
            self.dataloader_num_workers = self.num_workers


class DatasetType:
    """Dataset type constants for training."""

    INSTRUCTION = "instruction"
    CPT = "cpt"
    # DISABLED: DPO = "dpo"
    GRPO = "grpo"


@dataclass
class CPTConfig:
    """Configuration for Continued Pre-Training (CPT).

    CPT is used to continue pre-training on domain-specific data
    without instruction/output formatting.

    Args:
        dataset_type: Type of dataset - "instruction" or "cpt"
        text_key: Key for text field in JSON/JSONL files (for CPT)
        chat_template: Chat template to use for formatting
        mapping: Role mapping for chat templates
    """

    dataset_type: str = DatasetType.INSTRUCTION
    text_key: str = "text"
    chat_template: str = "alpaca"
    mapping: Optional[Dict[str, str]] = None


__all__ = [
    "InferenceConfig",
    "Prompts",
    "PEFTConfig",
    "TrainingConfig",
    "ScalableTrainingConfig",
    "DatasetType",
    "CPTConfig",
]
