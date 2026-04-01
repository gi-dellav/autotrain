"""DPO (Direct Preference Optimization) support for AutoTrain."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import InferenceConfig
from .core.model import Model
from .expert import Expert

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """
    Configuration for DPO training.

    Args:
        beta: Temperature parameter for DPO loss (default 0.1). Must be > 0.
        loss_type: Type of DPO loss ("sigmoid", "hinge", "ipo", "kto_pair")
        label_smoothing: Label smoothing factor (default 0.0). Must be in [0, 1).
        reference_free: If True, use model as its own reference
        f_divergence_type: Type of f-divergence for alignment (default "reverse_kl")
        reference_model_name: Optional reference model for DPO. If None, uses model itself.
        truncation_mode: How to truncate sequences ("keep_start", "keep_end")
        precompute_ref_log_probs: Precompute reference model log probs to save memory
        epochs: Number of training epochs. Must be > 0.
        batch_size: Per-device training batch size. Must be > 0.
        gradient_accumulation_steps: Number of gradient accumulation steps. Must be > 0.
        learning_rate: Training learning rate. Must be > 0.
        max_length: Maximum total sequence length (prompt + completion). Must be > 0.
        max_prompt_length: Maximum prompt length. Must be > 0 and <= max_length.
        generate_during_eval: If True, generate completions during evaluation
        is_encoder_decoder: Whether training an encoder-decoder model
    """

    beta: float = 0.1
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    reference_free: bool = False
    f_divergence_type: str = "reverse_kl"
    reference_model_name: Optional[str] = None
    truncation_mode: str = "keep_start"
    precompute_ref_log_probs: bool = False

    # Training parameters
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    max_length: int = 1024
    max_prompt_length: int = 512
    generate_during_eval: bool = False
    is_encoder_decoder: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.label_smoothing < 0 or self.label_smoothing >= 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}")
        if self.max_prompt_length <= 0:
            raise ValueError(f"max_prompt_length must be > 0, got {self.max_prompt_length}")
        if self.max_prompt_length > self.max_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) must be <= max_length ({self.max_length})"
            )
        valid_loss_types = {"sigmoid", "hinge", "ipo", "kto_pair"}
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'")
        valid_truncation_modes = {"keep_start", "keep_end"}
        if self.truncation_mode not in valid_truncation_modes:
            raise ValueError(
                f"truncation_mode must be one of {valid_truncation_modes}, got '{self.truncation_mode}'"
            )


@dataclass
class PreferenceSample:
    """
    A preference sample for DPO training.

    Contains a prompt with both a chosen (preferred) and rejected (less preferred) response.
    """

    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceSample":
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            metadata=data.get("metadata", {}),
        )


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization (DPO).

    DPO aligns models with human preferences by optimizing a simple loss
    based on pairs of chosen/rejected responses.

    Usage:
        model = Model("unsloth/Qwen3.5-27B-GGUF")
        dpo_trainer = DPOTrainer(model, dpo_config)

        # Add preference data
        dpo_trainer.add_preference_sample(
            prompt="What is 2+2?",
            chosen="2+2=4",
            rejected="2+2=5"
        )

        # Train with DPO
        dpo_trainer.train()
    """

    def __init__(
        self,
        model: Model,
        dpo_config: Optional[DPOConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize DPO trainer.

        Args:
            model: The Model to train
            dpo_config: DPO configuration
            inference_config: Inference configuration
        """
        self.model = model
        self.dpo_config = dpo_config or DPOConfig()
        self.inference_config = inference_config or InferenceConfig()

        # Preference dataset
        self._preference_samples: List[PreferenceSample] = []

        # Training state
        self._is_trained = False
        self._training_history: List[Dict[str, float]] = []
        self._output_dir: Optional[str] = None

        # Reference model (if using separate reference)
        self._reference_model = None

    def add_preference_sample(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a preference sample.

        Args:
            prompt: The input prompt
            chosen: The preferred (chosen) response
            rejected: The less preferred (rejected) response
            metadata: Optional metadata
        """
        sample = PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata or {},
        )
        self._preference_samples.append(sample)

    def add_preference_samples(
        self,
        samples: List[Tuple[str, str, str]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add multiple preference samples.

        Args:
            samples: List of (prompt, chosen, rejected) tuples
            metadata_list: Optional list of metadata dicts
        """
        if metadata_list is None:
            metadata_list = [{}] * len(samples)

        for (prompt, chosen, rejected), metadata in zip(samples, metadata_list):
            self.add_preference_sample(prompt, chosen, rejected, metadata)

    def generate_preference_samples(
        self,
        expert: Expert,
        prompts: List[str],
        count_per_prompt: int = 2,
    ) -> None:
        """
        Generate preference samples using an expert.

        The expert generates multiple responses per prompt, which are then
        ranked to create chosen/rejected pairs using pairwise comparisons.

        Args:
            expert: Expert model for generating responses
            prompts: List of prompts
            count_per_prompt: Number of responses to generate per prompt (default: 2)
        """
        for prompt in prompts:
            responses = []
            for _ in range(count_per_prompt):
                response = expert.solve(prompt)
                responses.append(response)

            if len(responses) < 2:
                continue

            # Generate all pairwise comparisons to create multiple preference pairs
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    comparison = expert.compare(prompt, responses[i], responses[j])

                    if comparison.get("winner") == "a":
                        chosen, rejected = responses[i], responses[j]
                    elif comparison.get("winner") == "b":
                        chosen, rejected = responses[j], responses[i]
                    else:
                        # Tie - skip this pair to avoid noisy data
                        continue

                    self.add_preference_sample(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        metadata={"generated": True, "pair": (i, j)},
                    )

    def create_from_rankings(
        self,
        prompt: str,
        responses: List[str],
        rankings: List[int],
    ) -> None:
        """
        Create preference samples from ranked responses.

        Args:
            prompt: The input prompt
            responses: List of responses
            rankings: List of ranks (lower = better)
        """
        if len(responses) != len(rankings):
            raise ValueError("responses and rankings must have same length")

        sorted_pairs = sorted(zip(responses, rankings), key=lambda x: x[1])

        best_response = sorted_pairs[0][0]
        for response, rank in sorted_pairs[1:]:
            if rank > sorted_pairs[0][1]:
                self.add_preference_sample(
                    prompt=prompt,
                    chosen=best_response,
                    rejected=response,
                    metadata={"ranking_based": True},
                )

    def export_dataset(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Export preference dataset to file.

        Args:
            path: Output file path
            format: Export format ("json" or "jsonl")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [sample.to_dict() for sample in self._preference_samples]

        if format == "jsonl":
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        logger.info("DPO dataset exported to %s (%d samples)", path, len(data))

    def import_dataset(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Import preference dataset from file.

        Args:
            path: Input file path
            format: Import format ("json" or "jsonl")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if format == "jsonl":
            with open(path) as f:
                data = [json.loads(line) for line in f]
        else:
            with open(path) as f:
                data = json.load(f)

        for item in data:
            sample = PreferenceSample.from_dict(item)
            self._preference_samples.append(sample)

        logger.info("Imported %d preference samples from %s", len(data), path)

    def train(
        self,
        eval_samples: Optional[List[PreferenceSample]] = None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the model using DPO.

        Args:
            eval_samples: Optional evaluation preference samples
            output_dir: Optional output directory for checkpoints
            resume_from_checkpoint: Resume from latest checkpoint if available

        Returns:
            Training summary dictionary
        """
        if not self._preference_samples:
            raise ValueError("No preference samples. Add samples before training.")

        try:
            from transformers import TrainingArguments
            from trl import DPOTrainer as TRLDPOTrainer

            try:
                from trl import DPOConfig as TRLDPOConfig
            except ImportError:
                TRLDPOConfig = None
        except ImportError as e:
            raise ImportError(
                f"Required package not found: {e}. Install with: pip install trl torch"
            )

        # Ensure model is loaded
        if not self.model.is_loaded:
            self.model.load_model()

        # Prepare dataset
        dataset = self._prepare_dataset()

        # Create reference model if needed
        ref_model = None
        if not self.dpo_config.reference_free and self.dpo_config.reference_model_name:
            try:
                from unsloth import FastLanguageModel

                ref_model, _ = FastLanguageModel.from_pretrained(
                    model_name=self.dpo_config.reference_model_name,
                    load_in_4bit=True,
                )
                ref_model = ref_model.model
            except ImportError:
                logger.warning(
                    "unsloth not available for reference model loading. "
                    "Falling back to reference_free mode."
                )
                ref_model = None

        # Training arguments
        if TRLDPOConfig:
            training_args = TRLDPOConfig(
                output_dir=output_dir or "./dpo_output",
                num_train_epochs=self.dpo_config.epochs,
                per_device_train_batch_size=self.dpo_config.batch_size,
                gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps,
                learning_rate=self.dpo_config.learning_rate,
                max_length=self.dpo_config.max_length,
                max_prompt_length=self.dpo_config.max_prompt_length,
                beta=self.dpo_config.beta,
                loss_type=self.dpo_config.loss_type,
                label_smoothing=self.dpo_config.label_smoothing,
                f_divergence_type=self.dpo_config.f_divergence_type,
                truncation_mode=self.dpo_config.truncation_mode,
                precompute_ref_log_probs=self.dpo_config.precompute_ref_log_probs,
                logging_steps=10,
                save_steps=100,
                eval_steps=100 if eval_samples else None,
                seed=42,
                remove_unused_columns=False,
            )
        else:
            training_args = TrainingArguments(
                output_dir=output_dir or "./dpo_output",
                num_train_epochs=self.dpo_config.epochs,
                per_device_train_batch_size=self.dpo_config.batch_size,
                gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps,
                learning_rate=self.dpo_config.learning_rate,
                logging_steps=10,
                save_steps=100,
                eval_steps=100 if eval_samples else None,
                seed=42,
                remove_unused_columns=False,
            )

        # Create DPO trainer
        trainer_kwargs = {
            "model": self.model.fast_model,
            "ref_model": ref_model,
            "args": training_args,
            "train_dataset": dataset,
            "eval_dataset": self._prepare_dataset(eval_samples) if eval_samples else None,
            "tokenizer": self.model.tokenizer,
            "max_length": self.dpo_config.max_length,
            "max_prompt_length": self.dpo_config.max_prompt_length,
        }

        if not TRLDPOConfig:
            trainer_kwargs.update(
                {
                    "beta": self.dpo_config.beta,
                    "loss_type": self.dpo_config.loss_type,
                    "label_smoothing": self.dpo_config.label_smoothing,
                }
            )

        # Wire reference_free: if True and no separate ref_model, pass ref_model=None
        if self.dpo_config.reference_free and ref_model is None:
            trainer_kwargs["ref_model"] = None

        trainer = TRLDPOTrainer(**trainer_kwargs)

        # Train
        logger.info("Starting DPO training with %d samples...", len(dataset))
        trainer.train()

        # Save the trained adapter
        final_output_dir = output_dir or "./dpo_output"
        self._output_dir = final_output_dir

        if hasattr(self.model.fast_model, "save_pretrained"):
            adapter_path = Path(final_output_dir) / "adapter"
            adapter_path.mkdir(parents=True, exist_ok=True)
            self.model.fast_model.save_pretrained(str(adapter_path))
            logger.info("DPO adapter saved to %s", adapter_path)

        self._is_trained = True

        return {
            "samples_used": len(dataset),
            "epochs": self.dpo_config.epochs,
            "trained": True,
            "output_dir": final_output_dir,
        }

    def _prepare_dataset(
        self,
        samples: Optional[List[PreferenceSample]] = None,
    ):
        """
        Prepare dataset for DPO training.

        Args:
            samples: Optional samples to use (default: self._preference_samples)

        Returns:
            HuggingFace Dataset
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package required. Install with: pip install datasets")

        samples = samples or self._preference_samples

        template = getattr(self.model, "_template", None)

        if template is not None:
            data = []
            for sample in samples:
                prompt_text = template.format_prompt(instruction=sample.prompt)

                chosen_text = template.assistant_template.format(output=sample.chosen)
                rejected_text = template.assistant_template.format(output=sample.rejected)

                data.append(
                    {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                    }
                )
        else:
            data = [
                {
                    "prompt": s.prompt,
                    "chosen": s.chosen,
                    "rejected": s.rejected,
                }
                for s in samples
            ]

        return Dataset.from_list(data)

    def get_samples(self) -> List[PreferenceSample]:
        """Get all preference samples."""
        return self._preference_samples.copy()

    def clear_samples(self) -> None:
        """Clear all preference samples."""
        self._preference_samples.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._preference_samples:
            return {"total_samples": 0}

        prompt_lengths = [len(s.prompt) for s in self._preference_samples]
        chosen_lengths = [len(s.chosen) for s in self._preference_samples]
        rejected_lengths = [len(s.rejected) for s in self._preference_samples]

        return {
            "total_samples": len(self._preference_samples),
            "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "avg_chosen_length": sum(chosen_lengths) / len(chosen_lengths),
            "avg_rejected_length": sum(rejected_lengths) / len(rejected_lengths),
            "max_prompt_length": max(prompt_lengths),
            "max_chosen_length": max(chosen_lengths),
        }


def train_dpo(
    model: Model,
    preference_samples: List[PreferenceSample],
    dpo_config: Optional[DPOConfig] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for DPO training.

    Args:
        model: Model to train
        preference_samples: List of preference samples
        dpo_config: DPO configuration
        output_dir: Output directory

    Returns:
        Training summary
    """
    trainer = DPOTrainer(model, dpo_config)

    for sample in preference_samples:
        trainer.add_preference_sample(
            prompt=sample.prompt,
            chosen=sample.chosen,
            rejected=sample.rejected,
            metadata=sample.metadata,
        )

    return trainer.train(output_dir=output_dir)
