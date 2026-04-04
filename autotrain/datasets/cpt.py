"""Continued Pre-Training (CPT) dataset utilities."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from autotrain.data_types import Sample


@dataclass
class CPTDatasetConfig:
    """Configuration for CPT dataset."""

    name: str = "cpt_dataset"
    text_key: str = "text"
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42


class CPTDataset:
    """Dataset class for Continued Pre-Training (CPT).

    CPT datasets contain raw text for continued pre-training,
    without instruction/output formatting. This is useful for:
    - Domain adaptation
    - Knowledge distillation from raw text
    - Continuation training on specific corpora

    Supports:
    - Raw .txt files (one document per line)
    - JSON/JSONL files with 'text' field
    - List of text strings
    """

    def __init__(
        self,
        name: str = "cpt_dataset",
        text_key: str = "text",
    ):
        """Initialize CPTDataset.

        Args:
            name: Dataset name identifier
            text_key: Key for text field in JSON/JSONL format
        """
        self.name = name
        self.text_key = text_key
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

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> "CPTDataset":
        """Add a single text sample.

        For CPT, we treat the entire text as the input_data,
        with empty output_data (model continues from input).
        """
        sample = Sample(
            input_data=text,
            output_data="",
            metadata=metadata or {"type": "cpt"},
        )
        self._samples.append(sample)
        self._updated_at = datetime.now().isoformat()
        return self

    def add_texts(self, texts: Union[List[str], List[Dict[str, Any]]]) -> "CPTDataset":
        """Add multiple text samples.

        Args:
            texts: List of strings or list of dicts with text_key field
        """
        for item in texts:
            if isinstance(item, str):
                self.add_text(item)
            elif isinstance(item, dict):
                text = str(item.get(self.text_key, item.get("text", "")))
                metadata = {k: v for k, v in item.items() if k != self.text_key}
                self.add_text(text, metadata)
        return self

    @classmethod
    def from_text_file(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        text_key: str = "text",
    ) -> "CPTDataset":
        """Load CPT dataset from a text file.

        Each line in the file is treated as a separate document.

        Args:
            path: Path to .txt file
            name: Optional dataset name
            text_key: Key for text field (not used for txt files)

        Returns:
            CPTDataset instance
        """
        path = Path(path)
        dataset = cls(name=name or path.stem, text_key=text_key)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.add_text(line, {"source": str(path)})

        return dataset

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        text_key: str = "text",
    ) -> "CPTDataset":
        """Load CPT dataset from JSON file.

        Args:
            path: Path to .json file
            name: Optional dataset name
            text_key: Key for text field in JSON

        Returns:
            CPTDataset instance
        """
        path = Path(path)
        dataset = cls(name=name or path.stem, text_key=text_key)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            dataset.add_texts(data)
        elif isinstance(data, dict):
            if "text" in data:
                dataset.add_text(data["text"])
            elif text_key in data:
                dataset.add_text(data[text_key])

        return dataset

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        text_key: str = "text",
    ) -> "CPTDataset":
        """Load CPT dataset from JSONL file.

        Args:
            path: Path to .jsonl file
            name: Optional dataset name
            text_key: Key for text field in JSONL

        Returns:
            CPTDataset instance
        """
        path = Path(path)
        dataset = cls(name=name or path.stem, text_key=text_key)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    text = item.get(text_key, item.get("text", ""))
                    if text:
                        metadata = {k: v for k, v in item.items() if k != text_key and k != "text"}
                        dataset.add_text(text, metadata)

        return dataset

    @classmethod
    def from_list(
        cls,
        texts: List[str],
        name: str = "cpt_dataset",
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> "CPTDataset":
        """Create CPT dataset from a list of text strings.

        Args:
            texts: List of text strings
            name: Dataset name
            metadata: Optional list of metadata dicts for each text

        Returns:
            CPTDataset instance
        """
        dataset = cls(name=name)
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else {"index": i}
            dataset.add_text(text, meta)
        return dataset

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        text_key: str = "text",
    ) -> "CPTDataset":
        """Load CPT dataset from file (auto-detect format).

        Args:
            path: Path to file (.txt, .json, .jsonl)
            name: Optional dataset name
            text_key: Key for text field

        Returns:
            CPTDataset instance
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return cls.from_text_file(path, name, text_key)
        elif suffix == ".json":
            return cls.from_json(path, name, text_key)
        elif suffix == ".jsonl":
            return cls.from_jsonl(path, name, text_key)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def save(
        self,
        path: Union[str, Path],
        format: str = "jsonl",
    ) -> None:
        """Save CPT dataset to file.

        Args:
            path: Output path
            format: Output format ("json", "jsonl", "text")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        texts = [{"text": s.input_data, **s.metadata} for s in self._samples]

        if format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for item in texts:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif format == "text":
            with open(path, "w", encoding="utf-8") as f:
                for s in self._samples:
                    f.write(s.input_data + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)

        print(f"CPT dataset saved to {path} ({len(texts)} samples)")

    def to_list(self) -> List[str]:
        """Get list of text samples."""
        return [s.input_data for s in self._samples]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._samples:
            return {"total_samples": 0, "total_chars": 0}

        texts = [s.input_data for s in self._samples]
        char_counts = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]

        return {
            "total_samples": len(self._samples),
            "total_chars": sum(char_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "total_words": sum(word_counts),
            "avg_words": sum(word_counts) / len(word_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "created_at": self._created_at,
            "updated_at": self._updated_at,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __repr__(self) -> str:
        return f"CPTDataset(name='{self.name}', samples={len(self._samples)})"


__all__ = ["CPTDataset", "CPTDatasetConfig"]
