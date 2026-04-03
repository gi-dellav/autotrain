"""Utility for dynamic parameter values based on iteration count."""

import math
from typing import Callable, Optional, Union


def validate_temperature(value: float) -> None:
    """Validate that a temperature value is within acceptable range."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"Temperature must be a number, got {type(value).__name__}")
    if value < 0.0 or value > 2.0:
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {value}")


def validate_epochs(value: float) -> None:
    """Validate that an epochs value is a positive integer."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"Epochs must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Epochs must be positive, got {value}")
    if math.floor(value) != value:
        raise ValueError(f"Epochs must be an integer, got {value}")


def validate_learning_rate(value: float) -> None:
    """Validate that a learning rate value is positive."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"Learning rate must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Learning rate must be positive, got {value}")


def validate_lora_alpha(value: float) -> None:
    """Validate that a LoRA alpha value is positive."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"LoRA alpha must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {value}")


def validate_lora_dropout(value: float) -> None:
    """Validate that a LoRA dropout value is in valid range."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"LoRA dropout must be a number, got {type(value).__name__}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"LoRA dropout must be between 0.0 and 1.0, got {value}")


def validate_dpo_beta(value: float) -> None:
    """Validate that a DPO beta value is positive."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"DPO beta must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"DPO beta must be positive, got {value}")


def validate_weight_decay(value: float) -> None:
    """Validate that a weight decay value is non-negative."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"Weight decay must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"Weight decay must be non-negative, got {value}")


def validate_batch_size(value: int) -> None:
    """Validate that a batch size value is a positive integer."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"Batch size must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"Batch size must be positive, got {value}")
    if math.floor(value) != value:
        raise ValueError(f"Batch size must be an integer, got {value}")


def validate_lora_rank(value: int) -> None:
    """Validate that a LoRA rank value is a positive integer."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"LoRA rank must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"LoRA rank must be positive, got {value}")
    if math.floor(value) != value:
        raise ValueError(f"LoRA rank must be an integer, got {value}")


def evaluate_temperature(
    temperature: Optional[Union[float, Callable[[int], float]]],
    iteration: int,
) -> float:
    """Evaluate temperature for the given iteration."""
    if temperature is None:
        return 0.7
    if callable(temperature):
        value = temperature(iteration)
    else:
        value = float(temperature)
    validate_temperature(value)
    return value


def evaluate_epochs(
    epochs: Optional[Union[int, Callable[[int], int]]],
    iteration: int,
) -> int:
    """Evaluate epochs for the given iteration."""
    if epochs is None:
        return 1
    if callable(epochs):
        value = epochs(iteration)
    else:
        value = int(epochs)
    validate_epochs(value)
    return int(value)


def evaluate_learning_rate(
    learning_rate: Optional[Union[float, Callable[[int], float]]],
    iteration: int,
) -> float:
    """Evaluate learning rate for the given iteration."""
    if learning_rate is None:
        return 1e-5
    if callable(learning_rate):
        value = learning_rate(iteration)
    else:
        value = float(learning_rate)
    validate_learning_rate(value)
    return value


def evaluate_lora_alpha(
    lora_alpha: Optional[Union[int, Callable[[int], int]]],
    iteration: int,
) -> int:
    """Evaluate LoRA alpha for the given iteration."""
    if lora_alpha is None:
        return 128
    if callable(lora_alpha):
        value = lora_alpha(iteration)
    else:
        value = int(lora_alpha)
    validate_lora_alpha(value)
    return int(value)


def evaluate_lora_dropout(
    lora_dropout: Optional[Union[float, Callable[[int], float]]],
    iteration: int,
) -> float:
    """Evaluate LoRA dropout for the given iteration."""
    if lora_dropout is None:
        return 0.0
    if callable(lora_dropout):
        value = lora_dropout(iteration)
    else:
        value = float(lora_dropout)
    validate_lora_dropout(value)
    return value


def evaluate_dpo_beta(
    beta: Optional[Union[float, Callable[[int], float]]],
    iteration: int,
) -> float:
    """Evaluate DPO beta for the given iteration."""
    if beta is None:
        return 0.1
    if callable(beta):
        value = beta(iteration)
    else:
        value = float(beta)
    validate_dpo_beta(value)
    return value


def evaluate_weight_decay(
    weight_decay: Optional[Union[float, Callable[[int], float]]],
    iteration: int,
) -> float:
    """Evaluate weight decay for the given iteration."""
    if weight_decay is None:
        return 0.01
    if callable(weight_decay):
        value = weight_decay(iteration)
    else:
        value = float(weight_decay)
    validate_weight_decay(value)
    return value


def evaluate_batch_size(
    batch_size: Optional[Union[int, Callable[[int], int]]],
    iteration: int,
) -> int:
    """Evaluate batch size for the given iteration."""
    if batch_size is None:
        return 2
    if callable(batch_size):
        value = batch_size(iteration)
    else:
        value = int(batch_size)
    validate_batch_size(value)
    return int(value)


def evaluate_lora_rank(
    lora_rank: Optional[Union[int, Callable[[int], int]]],
    iteration: int,
) -> int:
    """Evaluate LoRA rank for the given iteration."""
    if lora_rank is None:
        return 16
    if callable(lora_rank):
        value = lora_rank(iteration)
    else:
        value = int(lora_rank)
    validate_lora_rank(value)
    return int(value)


__all__ = [
    "validate_temperature",
    "validate_epochs",
    "validate_learning_rate",
    "validate_lora_alpha",
    "validate_lora_dropout",
    "validate_dpo_beta",
    "validate_weight_decay",
    "validate_batch_size",
    "validate_lora_rank",
    "evaluate_temperature",
    "evaluate_epochs",
    "evaluate_learning_rate",
    "evaluate_lora_alpha",
    "evaluate_lora_dropout",
    "evaluate_dpo_beta",
    "evaluate_weight_decay",
    "evaluate_batch_size",
    "evaluate_lora_rank",
]
