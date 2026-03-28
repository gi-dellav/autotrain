"""Dataset classes for AutoTrain training data management."""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from autotrain.data_types import Sample

if TYPE_CHECKING:
    from autotrain.expert import Expert
    from autotrain.datasets.cpt import CPTDataset


@dataclass
class DatasetConfig:
    """
    Configuration for AutoTrain dataset.

    Args:
        name: Dataset name identifier
        instruction_key: Key for instruction in dict format
        input_key: Key for additional input in dict format
        output_key: Key for output in dict format
        metadata_keys: Keys to preserve in metadata
        max_samples: Maximum number of samples to keep (None for unlimited)
        shuffle: Whether to shuffle samples
        seed: Random seed for shuffling
    """

    name: str = "default"
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    metadata_keys: List[str] = field(default_factory=lambda: ["source", "iteration"])
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42


class AutoTrainDataset:
    """
    Dataset class for managing AutoTrain training data.

    Provides:
    - Sample storage and management
    - Template-based formatting
    - Batch generation
    - Export/import (JSON, JSONL, Parquet)
    - Train/validation split
    - Sample filtering and transformation
    """

    def __init__(
        self,
        name: str = "default",
        config: Optional[DatasetConfig] = None,
        template: Optional[Any] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize AutoTrainDataset.

        Args:
            name: Dataset name
            config: Dataset configuration
            template: Instruction template for formatting
            model_name: Model name for auto-detecting template
        """
        self.name = name
        self.config = config or DatasetConfig(name=name)

        # Auto-detect template if model_name provided
        if template:
            self.template = template
        elif model_name:
            from autotrain.templates import auto_detect_template

            self.template = auto_detect_template(model_name)
        else:
            self.template = None

        self._samples: List[Sample] = []
        self._created_at = datetime.now().isoformat()
        self._updated_at = self._created_at

    @property
    def samples(self) -> List[Sample]:
        """Get all samples."""
        return self._samples.copy()

    @property
    def sample_count(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    @property
    def is_empty(self) -> bool:
        """Check if dataset is empty."""
        return len(self._samples) == 0

    def add_sample(
        self,
        input_data: str,
        output_data: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AutoTrainDataset":
        """Add a single sample."""
        sample = Sample(
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
        )
        self._samples.append(sample)
        self._updated_at = datetime.now().isoformat()
        return self

    def add_samples(
        self,
        samples: Union[List[Sample], List[Dict[str, Any]]],
    ) -> "AutoTrainDataset":
        """Add multiple samples."""
        for item in samples:
            if isinstance(item, Sample):
                self._samples.append(item)
            elif isinstance(item, dict):
                sample = Sample(
                    input_data=item.get(self.config.instruction_key, "")
                    or item.get("input_data", ""),
                    output_data=item.get(self.config.output_key, "") or item.get("output_data", ""),
                    metadata={k: item.get(k, {}) for k in self.config.metadata_keys},
                )
                self._samples.append(sample)

        self._updated_at = datetime.now().isoformat()
        return self

    def remove_sample(self, index: int) -> bool:
        """Remove a sample by index."""
        if 0 <= index < len(self._samples):
            self._samples.pop(index)
            self._updated_at = datetime.now().isoformat()
            return True
        return False

    def clear_samples(self) -> "AutoTrainDataset":
        """Clear all samples."""
        self._samples.clear()
        self._updated_at = datetime.now().isoformat()
        return self

    def filter(
        self,
        predicate: Callable[[Sample], bool],
    ) -> "AutoTrainDataset":
        """Filter samples based on predicate."""
        filtered = AutoTrainDataset(
            name=f"{self.name}_filtered",
            config=self.config,
            template=self.template,
        )
        filtered._samples = [s for s in self._samples if predicate(s)]
        return filtered

    def transform(
        self,
        transform_fn: Callable[[Sample], Sample],
    ) -> "AutoTrainDataset":
        """Transform all samples."""
        self._samples = [transform_fn(s) for s in self._samples]
        self._updated_at = datetime.now().isoformat()
        return self

    def train_test_split(
        self,
        test_size: float = 0.2,
        seed: Optional[int] = None,
    ) -> tuple["AutoTrainDataset", "AutoTrainDataset"]:
        """Split dataset into train and test sets."""
        if not 0.0 <= test_size <= 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")

        indices = list(range(len(self._samples)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_dataset = AutoTrainDataset(
            name=f"{self.name}_train",
            config=self.config,
            template=self.template,
        )
        train_dataset._samples = [self._samples[i] for i in train_indices]

        test_dataset = AutoTrainDataset(
            name=f"{self.name}_test",
            config=self.config,
            template=self.template,
        )
        test_dataset._samples = [self._samples[i] for i in test_indices]

        return train_dataset, test_dataset

    def format_sample(
        self,
        sample: Sample,
        include_output: bool = True,
    ) -> str:
        """Format a sample using the template."""
        if self.template:
            if include_output:
                return self.template.format_training_sample(
                    instruction=sample.input_data,
                    output=sample.output_data,
                )
            else:
                return self.template.format_prompt(
                    instruction=sample.input_data,
                )
        else:
            # Default format
            if include_output:
                return (
                    f"### Instruction:\n{sample.input_data}\n\n### Response:\n{sample.output_data}"
                )
            else:
                return f"### Instruction:\n{sample.input_data}\n\n### Response:\n"

    def format_all(
        self,
        include_output: bool = True,
    ) -> List[str]:
        """Format all samples."""
        return [self.format_sample(s, include_output) for s in self._samples]

    def get_training_format(self) -> List[Dict[str, str]]:
        """
        Get samples in training-ready format (input/output pairs).

        Returns:
            List of dictionaries with 'input' and 'output' keys
        """
        return [{"input": s.input_data, "output": s.output_data} for s in self._samples]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert samples to list of dictionaries."""
        return [
            {
                self.config.instruction_key: s.input_data,
                self.config.input_key: "",
                self.config.output_key: s.output_data,
                **s.metadata,
            }
            for s in self._samples
        ]

    def to_huggingface_dataset(self) -> Any:
        """Convert to HuggingFace Dataset."""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package required. Install with: pip install datasets")

        data = self.to_dict_list()
        return Dataset.from_list(data)

    def save(
        self,
        path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """Save dataset to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict_list()

        if format == "jsonl":
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        elif format == "parquet":
            try:
                import pandas as pd

                df = pd.DataFrame(data)
                df.to_parquet(path, index=False)
            except ImportError:
                raise ImportError(
                    "pandas and pyarrow required for Parquet. "
                    "Install with: pip install pandas pyarrow"
                )
        else:  # json
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        print(f"Dataset saved to {path} ({len(data)} samples)")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        format: Optional[str] = None,
        template: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> "AutoTrainDataset":
        """Load dataset from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Auto-detect format
        if format is None:
            suffix = path.suffix.lower()
            format_map = {
                ".json": "json",
                ".jsonl": "jsonl",
                ".parquet": "parquet",
            }
            format = format_map.get(suffix, "json")

        # Load data
        if format == "jsonl":
            with open(path) as f:
                data = [json.loads(line) for line in f]
        elif format == "parquet":
            try:
                import pandas as pd

                df = pd.read_parquet(path)
                data = df.to_dict("records")
            except ImportError:
                raise ImportError("pandas and pyarrow required for Parquet")
        else:  # json
            with open(path) as f:
                data = json.load(f)

        # Create dataset
        dataset = cls(
            name=name or path.stem,
            template=template,
            model_name=model_name,
        )
        dataset.add_samples(data)

        print(f"Loaded {len(dataset._samples)} samples from {path}")
        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._samples:
            return {
                "total_samples": 0,
                "avg_input_length": 0,
                "avg_output_length": 0,
            }

        input_lengths = [len(s.input_data.split()) for s in self._samples]
        output_lengths = [len(s.output_data.split()) for s in self._samples]

        return {
            "total_samples": len(self._samples),
            "avg_input_length": sum(input_lengths) / len(input_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "max_input_length": max(input_lengths),
            "max_output_length": max(output_lengths),
            "min_input_length": min(input_lengths),
            "min_output_length": min(output_lengths),
            "created_at": self._created_at,
            "updated_at": self._updated_at,
        }

    @classmethod
    def from_cpt_dataset(cls, cpt_dataset: "CPTDataset") -> "AutoTrainDataset":
        """Create AutoTrainDataset from CPTDataset.

        Args:
            cpt_dataset: CPTDataset instance

        Returns:
            AutoTrainDataset with samples converted from CPT format
        """
        dataset = cls(name=f"{cpt_dataset.name}_converted")
        for sample in cpt_dataset.samples:
            dataset._samples.append(
                Sample(
                    input_data=sample.input_data,
                    output_data=sample.output_data,
                    metadata={**sample.metadata, "dataset_type": "cpt"},
                )
            )
        return dataset

    def distill(
        self,
        expert: "Expert",
        task: Optional[str] = None,
        count: int = 10,
    ) -> "AutoTrainDataset":
        """Generate training samples through expert distillation.

        Args:
            expert: Expert instance to use for generation
            task: Task description for generation
            count: Number of samples to generate

        Returns:
            New AutoTrainDataset with distilled samples
        """
        from autotrain.datasets.distill import ExpertDistiller

        distiller = ExpertDistiller(expert=expert)
        samples = distiller.generate(task=task or self.name, count=count)

        new_dataset = AutoTrainDataset(
            name=f"{self.name}_distilled",
            template=self.template,
        )
        new_dataset._samples = samples
        return new_dataset

    def distill_from_inputs(
        self,
        inputs: List[str],
        expert: Optional["Expert"] = None,
        model_name: str = "deepseek/deepseek-v3.2",
        api_key: Optional[str] = None,
    ) -> "AutoTrainDataset":
        """Generate outputs for existing inputs using expert distillation.

        Args:
            inputs: List of input prompts
            expert: Optional Expert instance (creates one if not provided)
            model_name: Model name if creating new Expert
            api_key: Optional API key

        Returns:
            New AutoTrainDataset with input-output pairs
        """
        from autotrain.datasets.distill import ExpertDistiller

        if expert is None:
            expert = Expert(model_name=model_name, api_key=api_key)

        distiller = ExpertDistiller(expert=expert)
        samples = distiller.distill_from_inputs(inputs)

        new_dataset = AutoTrainDataset(
            name=f"{self.name}_distilled",
            template=self.template,
        )
        new_dataset._samples = samples
        return new_dataset

    def to_cpt_format(self) -> List[Dict[str, Any]]:
        """Convert samples to CPT format (text only, no instruction/output).

        Returns:
            List of dicts with 'text' key for each sample
        """
        texts = []
        for s in self._samples:
            text = s.input_data
            if s.output_data:
                text = f"{text} {s.output_data}"
            texts.append({"text": text, **s.metadata})
        return texts

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        """Get sample by index."""
        return self._samples[idx]

    def __iter__(self):
        """Iterate over samples."""
        return iter(self._samples)

    def __repr__(self) -> str:
        """String representation."""
        return f"AutoTrainDataset(name='{self.name}', samples={len(self._samples)})"


__all__ = ["AutoTrainDataset", "DatasetConfig"]
