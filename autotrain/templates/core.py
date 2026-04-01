"""Instruction tuning templates for AutoTrain."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TemplateType(Enum):
    """Pre-defined template types."""

    ALPACA = "alpaca"
    CHATML = "chatml"
    LLAMA3 = "llama3"
    LLAMA31 = "llama-3.1"
    LLAMA32 = "llama-3.2"
    LLAMA33 = "llama-3.3"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    GEMMA3 = "gemma3"
    PHI = "phi"
    PHI3 = "phi-3"
    PHI35 = "phi-3.5"
    PHI4 = "phi-4"
    QWEN25 = "qwen-2.5"
    ZEPHYR = "zephyr"
    VICUNA = "vicuna"
    VICUNA_OLD = "vicuna_old"
    CUSTOM = "custom"


@dataclass
class InstructionTemplate:
    """
    Template for formatting instruction tuning data.

    Supports system prompt, user input, and assistant response formatting.
    """

    name: str
    template_type: TemplateType = TemplateType.CUSTOM

    # Template components
    system_prompt: str = ""
    system_template: str = "{system}"
    user_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    assistant_template: str = "{output}"

    # Formatting
    separator: str = "\n"
    eos_token: str = ""

    # Metadata
    description: str = ""

    def format(
        self,
        instruction: str = "",
        input_data: str = "",
        output: str = "",
        system_prompt: Optional[str] = None,
        include_output: bool = True,
    ) -> str:
        """Format a sample using this template."""
        parts = []

        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            parts.append(self.system_template.format(system=sys_prompt))

        user_text = self.user_template.format(
            instruction=instruction,
            input=input_data,
        )
        parts.append(user_text)

        if include_output and output:
            parts.append(self.assistant_template.format(output=output))

        return self.separator.join(parts)

    def format_prompt(
        self,
        instruction: str = "",
        input_data: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format a prompt for inference (without output)."""
        return self.format(
            instruction=instruction,
            input_data=input_data,
            output="",
            system_prompt=system_prompt,
            include_output=False,
        )

    def format_training_sample(
        self,
        instruction: str = "",
        input_data: str = "",
        output: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format a complete training sample."""
        return self.format(
            instruction=instruction,
            input_data=input_data,
            output=output,
            system_prompt=system_prompt,
            include_output=True,
        )


__all__ = ["InstructionTemplate", "TemplateType"]
