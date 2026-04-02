"""Data types and sample classes for AutoTrain."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Sample:
    """Represents a training sample."""

    input_data: str
    output_data: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
        }

    def to_conversation(self) -> List[Dict[str, str]]:
        """
        Convert sample to conversation format.

        Returns:
            List of messages (User, Assistant)
        """
        return [
            {"role": "user", "content": self.input_data},
            {"role": "assistant", "content": self.output_data},
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create sample from dictionary."""
        return cls(
            input_data=data.get("input_data", ""),
            output_data=data.get("output_data", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisionSample(Sample):
    """
    Represents a vision-language training sample.

    Extends Sample with image support for multi-modal training.
    """

    images: List[Union[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "input_data": self.input_data,
            "output_data": self.output_data,
            "images": self.images,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionSample":
        """Create sample from dictionary."""
        return cls(
            input_data=data.get("input_data", ""),
            output_data=data.get("output_data", ""),
            images=data.get("images", []),
            metadata=data.get("metadata", {}),
        )

    def to_conversation(self) -> List[Dict[str, Any]]:
        """
        Convert to conversation format for vision models.

        Returns:
            List of messages (User, Assistant)
        """
        content = []
        for img in self.images:
            if isinstance(img, str):
                # Handle both URL and local path as URL for convenience
                content.append({"type": "image_url", "image_url": {"url": img}})
            else:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": self.input_data})

        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": self.output_data}]},
        ]


__all__ = ["Sample", "VisionSample"]
