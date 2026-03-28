"""Vision Data Collator for VLM fine-tuning."""

from typing import Any, Callable, Dict, List, Optional, Union


class VisionDataCollator:
    """
    Data collator for vision-language model training.

    Handles multi-modal tokenization and formatting for VLMs.
    Similar to Unsloth's UnslothVisionDataCollator.

    Example usage:
        collator = VisionDataCollator(
            processor=tokenizer,
            max_seq_length=2048,
            train_on_responses_only=True,
        )
    """

    def __init__(
        self,
        processor: Any,
        max_seq_length: Optional[int] = None,
        formatting_func: Optional[Callable] = None,
        resize: Union[str, tuple] = "min",
        ignore_index: int = -100,
        train_on_responses_only: bool = False,
        instruction_part: Optional[str] = None,
        response_part: Optional[str] = None,
        force_match: bool = True,
        completion_only_loss: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        resize_dimension: Union[int, str] = 0,
        snap_to_patch_size: bool = False,
    ):
        """
        Initialize the Vision Data Collator.

        Args:
            processor: The tokenizer/processor for the model
            max_seq_length: Maximum sequence length
            formatting_func: Function for transforming the text
            resize: Resize strategy - "min", "max", or (width, height) tuple
            ignore_index: Index to ignore in loss calculation
            train_on_responses_only: Only train on assistant responses
            instruction_part: Instruction part marker for train_on_responses_only
            response_part: Response part marker for train_on_responses_only
            force_match: Match newlines as well
            completion_only_loss: Ignore padding vision tokens
            pad_to_multiple_of: Pad to multiple of this value
            resize_dimension: Dimension to resize (0, 1, 'max', 'min')
            snap_to_patch_size: Force image to be multiple of patch size
        """
        self.processor = processor
        self.max_seq_length = max_seq_length or 2048
        self.formatting_func = formatting_func
        self.resize = resize
        self.ignore_index = ignore_index
        self.train_on_responses_only = train_on_responses_only
        self.instruction_part = instruction_part
        self.response_part = response_part
        self.force_match = force_match
        self.completion_only_loss = completion_only_loss
        self.pad_to_multiple_of = pad_to_multiple_of
        self.resize_dimension = resize_dimension
        self.snap_to_patch_size = snap_to_patch_size

    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        """
        Collate features into a batch.

        Args:
            features: List of feature dictionaries

        Returns:
            Collated batch
        """
        if not features:
            return {}

        first = features[0]
        batch = {}

        if "pixel_values" in first:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

        if "input_ids" in first:
            batch["input_ids"] = torch.tensor([f["input_ids"] for f in features])
            batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features])

        if "labels" in first:
            batch["labels"] = torch.tensor([f["labels"] for f in features])

        return batch

    def process_sample(self, sample: Dict) -> Dict[str, Any]:
        """
        Process a single sample.

        Args:
            sample: Sample dictionary with messages and/or images

        Returns:
            Processed sample
        """
        messages = sample.get("messages", [])

        if not messages:
            return {}

        try:
            from transformers import AutoProcessor

            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )

                inputs = self.processor(
                    text=text,
                    images=sample.get("images"),
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                )
            else:
                inputs = self._process_basic(sample)

            if self.train_on_responses_only:
                inputs = self._mask_instruction(inputs, messages)

            return inputs
        except Exception as e:
            print(f"Warning: Error processing sample: {e}")
            return self._process_basic(sample)

    def _process_basic(self, sample: Dict) -> Dict[str, Any]:
        """Basic processing without chat template."""
        messages = sample.get("messages", [])
        images = sample.get("images", [])

        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(f"{role}: {item.get('text', '')}")
            else:
                text_parts.append(f"{role}: {content}")

        text = "\n".join(text_parts)

        inputs = self.processor(
            text=text,
            images=images[0] if images else None,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
        )

        return inputs

    def _mask_instruction(self, inputs: Dict, messages: List[Dict]) -> Dict[str, Any]:
        """Mask non-response tokens for train_on_responses_only."""
        if "labels" not in inputs:
            return inputs

        labels = inputs["labels"].clone()

        if self.instruction_part and self.response_part:
            text = self.processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)

            for i, t in enumerate(text):
                instruction_end = t.find(self.instruction_part)
                response_start = t.find(self.response_part)

                if (
                    instruction_end != -1
                    and response_start != -1
                    and response_start > instruction_end
                ):
                    response_token_pos = 0
                    for j, token_id in enumerate(inputs["input_ids"][i]):
                        token_str = self.processor.decode(token_id)
                        if self.response_part in token_str or (
                            self.force_match and "\n" in token_str and response_start < j * 10
                        ):
                            response_token_pos = j
                            break

                    if response_token_pos > 0:
                        labels[i][:response_token_pos] = self.ignore_index

        inputs["labels"] = labels
        return inputs

    def format_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Format an entire dataset.

        Args:
            dataset: List of sample dictionaries

        Returns:
            List of formatted samples
        """
        formatted = []
        for sample in dataset:
            processed = self.process_sample(sample)
            if processed:
                formatted.append(processed)
        return formatted


__all__ = ["VisionDataCollator"]


try:
    import torch
except ImportError:
    pass
