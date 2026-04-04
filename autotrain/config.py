"""Configuration classes for AutoTrain."""

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union


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

    temperature: float = 0.4
    temperature_fn: Optional[Callable[[int], float]] = None
    max_tokens: int = 1024
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    thinking: bool = True
    max_workers: int = 8
    use_async: bool = False
    async_max_concurrent: int = 8
    producer_temperature: Union[float, Callable[[int], float]] = 1.0
    splitter_temperature: Union[float, Callable[[int], float]] = 0.3


@dataclass
class Prompts:
    """Configurable prompts for each component."""

    producer: Optional[Union[str, list[str]]] = None
    solver: Optional[Union[str, list[str]]] = None
    splitter: Optional[Union[str, list[str]]] = None
    checker: Optional[Union[str, list[str]]] = None

    def get_producer(self, topic: str, format: str = "json") -> str:
        from .prompts import bake_producer

        prompt = _get_prompt(self.producer)
        return prompt if prompt is not None else bake_producer(topic, format)

    def get_solver(self, topic: Optional[str] = None, context: str = "") -> str:
        from .prompts import bake_solver

        prompt = _get_prompt(self.solver)
        return prompt if prompt is not None else bake_solver(topic, context)

    def get_splitter(self) -> str:
        from .prompts import SPLITTER_DEFAULT

        prompt = _get_prompt(self.splitter)
        return prompt if prompt is not None else SPLITTER_DEFAULT

    def get_checker(self) -> str:
        from .prompts import CHECKER_DEFAULT

        prompt = _get_prompt(self.checker)
        return prompt if prompt is not None else CHECKER_DEFAULT


@dataclass
class PEFTConfig:
    """Configuration for PEFT/LoRA fine-tuning."""

    r: int = 16
    lora_rank_fn: Optional[Callable[[int], int]] = None
    lora_alpha: int = 32
    lora_alpha_fn: Optional[Callable[[int], int]] = None
    lora_dropout: float = 0.0
    lora_dropout_fn: Optional[Callable[[int], float]] = None
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    target_modules: Optional[list[str]] = None
    use_rslora: bool = False
    loftq_config: Optional[Dict[str, Any]] = None
    tuning_method: str = "qlora"  # "lora" or "qlora"

    def get_target_modules(self, model_type: str = "llama") -> list[str]:
        if self.target_modules:
            return self.target_modules
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __post_init__(self) -> None:
        if self.tuning_method not in ["lora", "qlora"]:
            raise ValueError(f"tuning_method must be 'lora' or 'qlora', got '{self.tuning_method}'")


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    epochs: int = 3
    epochs_fn: Optional[Callable[[int], int]] = None
    batch_size: int = 2
    batch_size_fn: Optional[Callable[[int], int]] = None
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    learning_rate_fn: Optional[Callable[[int], float]] = None
    weight_decay: float = 0.01
    weight_decay_fn: Optional[Callable[[int], float]] = None
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
    keep_last_n_iters: Optional[int] = None


@dataclass
class ScalableTrainingConfig:
    """Configuration for scalable training features."""

    gradient_checkpointing: str = "unsloth"
    mixed_precision: str = "bf16"
    batch_size_auto_tune: bool = False
    max_memory_mb: Optional[int] = None
    num_workers: int = 4
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    use_flash_attention: bool = False

    def __post_init__(self) -> None:
        if self.mixed_precision not in ["fp16", "bf16", "fp32"]:
            raise ValueError(
                f"mixed_precision must be 'fp16', 'bf16', or 'fp32', got '{self.mixed_precision}'"
            )
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.gradient_checkpointing not in [True, False, "unsloth"]:
            raise ValueError(
                f"gradient_checkpointing must be True, False, or 'unsloth', got '{self.gradient_checkpointing}'"
            )
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        if self.dataloader_num_workers == 4 and self.num_workers != 4:
            self.dataloader_num_workers = self.num_workers


class DatasetType:
    INSTRUCTION = "instruction"
    CPT = "cpt"
    GRPO = "grpo"


@dataclass
class CPTConfig:
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
