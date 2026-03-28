"""Pre-defined instruction templates."""

from autotrain.templates.core import InstructionTemplate, TemplateType


def get_alpaca_template() -> InstructionTemplate:
    """Get the Alpaca instruction tuning template."""
    return InstructionTemplate(
        name="alpaca",
        template_type=TemplateType.ALPACA,
        system_prompt="Below is an instruction that describes a task.",
        system_template="{system}",
        user_template="### Instruction: {instruction}\n\n### Response: ",
        assistant_template="{output}",
        separator="\n\n",
        description="Alpaca-style",
    )


def get_chatml_template() -> InstructionTemplate:
    """Get the ChatML template."""
    return InstructionTemplate(
        name="chatml",
        template_type=TemplateType.CHATML,
        system_template="[SYS]{system}[/SYS]",
        user_template="[USR]{instruction}[/USR][ASST]",
        assistant_template="{output}[/ASST]",
        separator="",
        description="ChatML format",
    )


def get_llama3_template() -> InstructionTemplate:
    """Get the Llama 3 template."""
    return InstructionTemplate(
        name="llama3",
        template_type=TemplateType.LLAMA3,
        system_template="<|system|>{system}<|end|>",
        user_template="<|user|>{instruction}<|end|><|assistant|>",
        assistant_template="{output}<|end|>",
        separator="",
        description="Llama 3 format",
    )


def get_mistral_template() -> InstructionTemplate:
    """Get the Mistral template."""
    return InstructionTemplate(
        name="mistral",
        template_type=TemplateType.MISTRAL,
        system_template="[SYS]{system}[/SYS]",
        user_template="[INST]{instruction}[/INST]",
        assistant_template="{output}",
        separator=" ",
        description="Mistral format",
    )


def get_gemma_template() -> InstructionTemplate:
    """Get the Gemma template."""
    return InstructionTemplate(
        name="gemma",
        template_type=TemplateType.GEMMA,
        system_template="<sys>{system}",
        user_template="<user>{instruction}\n<model>",
        assistant_template="{output}",
        separator="\n",
        description="Gemma format",
    )


def get_phi_template() -> InstructionTemplate:
    """Get the Phi template."""
    return InstructionTemplate(
        name="phi",
        template_type=TemplateType.PHI,
        system_template="<|system|>\n{system}",
        user_template="<|user|>\n{instruction}",
        assistant_template="<|assistant|>\n{output}",
        separator="\n",
        description="Phi format",
    )


def get_zephyr_template() -> InstructionTemplate:
    """Get the Zephyr template."""
    return InstructionTemplate(
        name="zephyr",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>\n{system}",
        user_template="<|user|>\n{instruction}",
        assistant_template="<|assistant|>\n{output}",
        separator="\n",
        description="Zephyr format",
    )


def get_vicuna_template() -> InstructionTemplate:
    """Get the Vicuna template."""
    return InstructionTemplate(
        name="vicuna",
        template_type=TemplateType.CUSTOM,
        system_template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n{system}",
        user_template="USER: {instruction}",
        assistant_template="ASSISTANT: {output}</s>",
        separator="\n",
        description="Vicuna format",
    )


def get_vicuna_old_template() -> InstructionTemplate:
    """Get the Vicuna (old) template."""
    return InstructionTemplate(
        name="vicuna_old",
        template_type=TemplateType.CUSTOM,
        system_template="",
        user_template="USER: {instruction}",
        assistant_template="ASSISTANT: {output}</s>",
        separator="\n",
        description="Vicuna old format",
    )


def get_gemma2_template() -> InstructionTemplate:
    """Get the Gemma 2 template."""
    return InstructionTemplate(
        name="gemma2",
        template_type=TemplateType.CUSTOM,
        system_template="<start_of_turn>system\n{system}<end_of_turn>",
        user_template="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        assistant_template="{output}<end_of_turn>",
        separator="",
        description="Gemma 2 format",
    )


def get_gemma3_template() -> InstructionTemplate:
    """Get the Gemma 3 template."""
    return InstructionTemplate(
        name="gemma3",
        template_type=TemplateType.CUSTOM,
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        user_template="<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_template="{output}<|eot_id|>",
        separator="",
        description="Gemma 3 format",
    )


def get_phi3_template() -> InstructionTemplate:
    """Get the Phi-3 template."""
    return InstructionTemplate(
        name="phi-3",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>\n{system}",
        user_template="<|user|>\n{instruction}",
        assistant_template="<|assistant|>\n{output}",
        separator="<|end|>\n",
        description="Phi-3 format",
    )


def get_phi35_template() -> InstructionTemplate:
    """Get the Phi-3.5 template."""
    return InstructionTemplate(
        name="phi-3.5",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>\n{system}",
        user_template="<|user|>\n{instruction}",
        assistant_template="<|assistant|>\n{output}",
        separator="<|end|>\n",
        description="Phi-3.5 format",
    )


def get_phi4_template() -> InstructionTemplate:
    """Get the Phi-4 template."""
    return InstructionTemplate(
        name="phi-4",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>\n{system}",
        user_template="<|user|>\n{instruction}",
        assistant_template="<|assistant|>\n{output}",
        separator="<|end|>\n",
        description="Phi-4 format",
    )


def get_qwen25_template() -> InstructionTemplate:
    """Get the Qwen 2.5 template."""
    return InstructionTemplate(
        name="qwen-2.5",
        template_type=TemplateType.CUSTOM,
        system_template="<|im_start|>system\n{system}<|im_end|>",
        user_template="<|im_start|>user\n{instruction}<|im_end|><|im_start|>assistant\n",
        assistant_template="{output}<|im_end|>",
        separator="",
        description="Qwen 2.5 format",
    )


def get_llama31_template() -> InstructionTemplate:
    """Get the Llama 3.1 template."""
    return InstructionTemplate(
        name="llama-3.1",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>{system}<|end|>",
        user_template="<|user|>{instruction}<|end|><|assistant|>",
        assistant_template="{output}<|end|>",
        separator="",
        description="Llama 3.1 format",
    )


def get_llama32_template() -> InstructionTemplate:
    """Get the Llama 3.2 template."""
    return InstructionTemplate(
        name="llama-3.2",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>{system}<|end_of_text|>",
        user_template="<|user|>{instruction}<|end_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_template="{output}<|end_of_text|>",
        separator="",
        description="Llama 3.2 format",
    )


def get_llama33_template() -> InstructionTemplate:
    """Get the Llama 3.3 template."""
    return InstructionTemplate(
        name="llama-3.3",
        template_type=TemplateType.CUSTOM,
        system_template="<|system|>{system}<|end|>",
        user_template="<|user|>{instruction}<|end|><|assistant|>",
        assistant_template="{output}<|end|>",
        separator="",
        description="Llama 3.3 format",
    )
