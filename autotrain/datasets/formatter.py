"""Dataset formatting utilities."""

from typing import Any, Dict, List, Optional, Union

from autotrain.data_types import Sample


class DatasetFormatter:
    """
    Utility class for formatting datasets with various templates.

    Provides batch formatting, template conversion, and tokenization helpers.
    """

    def __init__(
        self,
        template: Optional[Any] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize DatasetFormatter.

        Args:
            template: Instruction template to use
            model_name: Model name for auto-detecting template
        """
        from autotrain.templates import auto_detect_template, get_template

        if template:
            self.template = template
        elif model_name:
            self.template = auto_detect_template(model_name)
        else:
            self.template = get_template("alpaca")

    @classmethod
    def from_model_name(cls, model_name: str) -> "DatasetFormatter":
        """Create formatter from model name."""
        return cls(model_name=model_name)

    @classmethod
    def from_template_name(cls, template_name: str) -> "DatasetFormatter":
        """Create formatter from template name."""
        from autotrain.templates import get_template

        return cls(template=get_template(template_name))

    def format_sample(
        self,
        instruction: str,
        output: str,
        input_data: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format a single sample."""
        return self.template.format_training_sample(  # type: ignore[no-any-return]
            instruction=instruction,
            input_data=input_data,
            output=output,
            system_prompt=system_prompt,
        )

    def format_prompt(
        self,
        instruction: str,
        input_data: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format a prompt for inference (without output)."""
        return self.template.format_prompt(  # type: ignore[no-any-return]
            instruction=instruction,
            input_data=input_data,
            system_prompt=system_prompt,
        )

    def format_batch(
        self,
        samples: List[Dict[str, str]],
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Format a batch of samples."""
        formatted = []
        for sample in samples:
            text = self.template.format_training_sample(
                instruction=sample.get(instruction_key, ""),
                input_data=sample.get(input_key, ""),
                output=sample.get(output_key, ""),
                system_prompt=system_prompt,
            )
            formatted.append(text)
        return formatted

    def format_samples(
        self,
        samples: List[Sample],
        include_output: bool = True,
    ) -> List[str]:
        """Format a list of Sample objects."""
        formatted = []
        for sample in samples:
            if include_output:
                text = self.template.format_training_sample(
                    instruction=sample.input_data,
                    output=sample.output_data,
                )
            else:
                text = self.template.format_prompt(
                    instruction=sample.input_data,
                )
            formatted.append(text)
        return formatted

    def tokenize_batch(
        self,
        tokenizer: Any,
        texts: List[str],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """Tokenize a batch of texts."""
        return tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )  # type: ignore[no-any-return]

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = False,
        add_eos_token: bool = True,
        tokenize: bool = False,
        tokenizer: Any = None,
    ) -> Union[str, List[int]]:
        """Apply chat template to conversation messages.

        This method follows Unsloth's API for applying chat templates.
        Supports both ChatML format (role/content) and ShareGPT format (from/value).

        Args:
            messages: List of message dictionaries.
            add_generation_prompt: If True, adds assistant prefix after last user message
            add_eos_token: If True, adds EOS token after assistant messages
            tokenize: If True, returns token IDs instead of string
            tokenizer: HuggingFace/Unsloth tokenizer (required if tokenize=True)

        Returns:
            Formatted string with template applied, or list of token IDs
        """
        from autotrain.templates import apply_chat_template

        # Delegate to main apply_chat_template for consistency
        return apply_chat_template(
            messages=messages,
            template=self.template,
            add_generation_prompt=add_generation_prompt,
            add_eos_token=add_eos_token,
            tokenize=tokenize,
            tokenizer=tokenizer,
        )

    def convert_to_chatml(
        self,
        instruction: str,
        output: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Convert sample to ChatML format."""
        from autotrain.templates import get_template

        chatml = get_template("chatml")
        return chatml.format_training_sample(
            instruction=instruction,
            output=output,
            system_prompt=system_prompt,
        )

    def convert_to_llama3(
        self,
        instruction: str,
        output: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Convert sample to Llama 3 format."""
        from autotrain.templates import get_template

        llama3 = get_template("llama3")
        return llama3.format_training_sample(
            instruction=instruction,
            output=output,
            system_prompt=system_prompt,
        )

    def standardize_sharegpt(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        role_mapping: Optional[Dict[str, str]] = None,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Convert ShareGPT format to ChatML role/content format.

        Args:
            data: List of conversations in ShareGPT format
            role_mapping: Mapping from ShareGPT roles to standard roles.
                         Default: {"human": "user", "gpt": "assistant"}

        Returns:
            List of conversations in ChatML format

        Examples:
            >>> formatter = DatasetFormatter.from_model_name("meta-llama/Llama-3-8b")
            >>> data = [[{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi"}]]
            >>> standardized = formatter.standardize_sharegpt(data)
        """
        from autotrain.templates import standardize_sharegpt

        return standardize_sharegpt(data, role_mapping=role_mapping)

    def to_sharegpt(
        self,
        data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        role_mapping: Optional[Dict[str, str]] = None,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Convert ChatML format to ShareGPT format.

        Args:
            data: List of conversations in ChatML format
            role_mapping: Mapping from standard roles to ShareGPT roles.
                         Default: {"user": "human", "assistant": "gpt"}

        Returns:
            List of conversations in ShareGPT format
        """
        from autotrain.templates import to_sharegpt

        return to_sharegpt(data, role_mapping=role_mapping)

    def merge_conversations(
        self,
        conversations: List[List[Dict[str, Any]]],
        extension_length: int = 3,
        seed: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Merge single-turn conversations into multi-turn conversations.

        Args:
            conversations: List of conversations
            extension_length: Number of conversations to merge (default: 3)
            seed: Random seed for reproducibility

        Returns:
            List of merged multi-turn conversations
        """
        from autotrain.templates import merge_conversations

        return merge_conversations(conversations, extension_length=extension_length, seed=seed)


__all__ = ["DatasetFormatter"]
