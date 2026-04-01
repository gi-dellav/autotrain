"""Instruction tuning templates for AutoTrain.

This module provides chat template functionality compatible with Unsloth's API.
Supports ShareGPT, ChatML, and Alpaca formats with proper role mapping and EOS token handling.

See: https://unsloth.ai/docs/basics/chat-templates
"""

import random
from typing import Any, Dict, List, Optional, Union

from autotrain.templates.core import InstructionTemplate, TemplateType
from autotrain.templates.registry import (
    get_alpaca_template,
    get_chatml_template,
    get_gemma_template,
    get_gemma2_template,
    get_gemma3_template,
    get_llama3_template,
    get_llama31_template,
    get_llama32_template,
    get_llama33_template,
    get_mistral_template,
    get_phi_template,
    get_phi3_template,
    get_phi35_template,
    get_phi4_template,
    get_qwen25_template,
    get_vicuna_template,
    get_vicuna_old_template,
    get_zephyr_template,
)

# Template registry - accessible via CHAT_TEMPLATES constant
TEMPLATE_REGISTRY = {
    "alpaca": get_alpaca_template(),
    "chatml": get_chatml_template(),
    "llama3": get_llama3_template(),
    "llama-3.1": get_llama31_template(),
    "llama-3.2": get_llama32_template(),
    "llama-3.3": get_llama33_template(),
    "mistral": get_mistral_template(),
    "gemma": get_gemma_template(),
    "gemma2": get_gemma2_template(),
    "gemma3": get_gemma3_template(),
    "phi": get_phi_template(),
    "phi-3": get_phi3_template(),
    "phi-3.5": get_phi35_template(),
    "phi-4": get_phi4_template(),
    "qwen-2.5": get_qwen25_template(),
    "zephyr": get_zephyr_template(),
    "vicuna": get_vicuna_template(),
    "vicuna_old": get_vicuna_old_template(),
}

# Public constant for listing available templates (Unsloth-compatible API)
CHAT_TEMPLATES = TEMPLATE_REGISTRY


def get_template(name: str) -> InstructionTemplate:
    """Get a pre-defined template by name."""
    if name not in TEMPLATE_REGISTRY:
        available = ", ".join(TEMPLATE_REGISTRY.keys())
        raise ValueError(f"Template '{name}' not found. Available templates: {available}")
    return TEMPLATE_REGISTRY[name]


def auto_detect_template(model_name: str) -> InstructionTemplate:
    """Auto-detect template based on model name."""
    model_name_lower = model_name.lower()

    if "llama-3.3" in model_name_lower or "llama33" in model_name_lower:
        return get_llama33_template()
    elif "llama-3.2" in model_name_lower or "llama32" in model_name_lower:
        return get_llama32_template()
    elif "llama-3.1" in model_name_lower or "llama31" in model_name_lower:
        return get_llama31_template()
    elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return get_llama3_template()
    elif "mistral" in model_name_lower:
        return get_mistral_template()
    elif "gemma-3" in model_name_lower or "gemma3" in model_name_lower:
        return get_gemma3_template()
    elif "gemma2" in model_name_lower:
        return get_gemma2_template()
    elif "gemma" in model_name_lower:
        return get_gemma_template()
    elif "phi-4" in model_name_lower or "phi4" in model_name_lower:
        return get_phi4_template()
    elif "phi-3.5" in model_name_lower or "phi35" in model_name_lower:
        return get_phi35_template()
    elif "phi-3" in model_name_lower or "phi3" in model_name_lower:
        return get_phi3_template()
    elif "phi" in model_name_lower:
        return get_phi_template()
    elif "qwen2.5" in model_name_lower or "qwen-2.5" in model_name_lower:
        return get_qwen25_template()
    elif "chatml" in model_name_lower:
        return get_chatml_template()
    elif "zephyr" in model_name_lower:
        return get_zephyr_template()
    elif "vicuna" in model_name_lower:
        return get_vicuna_template()
    else:
        return get_alpaca_template()


def create_custom_template(
    name: str,
    system_template: str = "",
    user_template: str = "",
    assistant_template: str = "",
    system_prompt: str = "",
    separator: str = "\n",
    eos_token: str = "",
    description: str = "",
) -> InstructionTemplate:
    """Create a custom instruction template."""
    return InstructionTemplate(
        name=name,
        template_type=TemplateType.CUSTOM,
        system_prompt=system_prompt,
        system_template=system_template,
        user_template=user_template,
        assistant_template=assistant_template,
        separator=separator,
        eos_token=eos_token,
        description=description,
    )


def format_sample(
    instruction: str,
    output: str,
    input_data: str = "",
    template: Optional[InstructionTemplate] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Convenience function to format a single sample."""
    if template is None:
        if model_name:
            template = auto_detect_template(model_name)
        else:
            template = get_alpaca_template()

    return template.format_training_sample(
        instruction=instruction,
        input_data=input_data,
        output=output,
        system_prompt=system_prompt,
    )


def format_samples_batch(
    samples: list[dict],
    template: Optional[InstructionTemplate] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
) -> list[str]:
    """Format a batch of samples."""
    if template is None:
        if model_name:
            template = auto_detect_template(model_name)
        else:
            template = get_alpaca_template()

    formatted = []
    for sample in samples:
        text = template.format_training_sample(
            instruction=sample.get(instruction_key, ""),
            input_data=sample.get(input_key, ""),
            output=sample.get(output_key, ""),
            system_prompt=system_prompt,
        )
        formatted.append(text)

    return formatted


def apply_chat_template(
    messages: List[Dict[str, Any]],
    template: Optional[InstructionTemplate] = None,
    role_mapping: Optional[Dict[str, str]] = None,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    add_bos_token: bool = True,
    add_eos_token: bool = True,
    tokenizer: Any = None,
) -> Union[str, List[int]]:
    """Apply chat template to a list of messages.

    This function follows Unsloth's API for applying chat templates to conversations.
    Supports both ChatML format (role/content) and ShareGPT format (from/value).

    Args:
        messages: List of message dictionaries. Supports two formats:
            - ChatML: [{"role": "user", "content": "Hello"}, ...]
            - ShareGPT: [{"from": "human", "value": "Hello"}, ...]
        template: InstructionTemplate to use. If None, uses alpaca template.
        role_mapping: Optional mapping for role names (e.g., {"human": "user", "gpt": "assistant"})
        add_generation_prompt: If True, adds assistant prompt prefix after last user message.
                              Useful for inference to prompt the model to start generating.
        tokenize: If True, returns token IDs instead of string (requires tokenizer argument).
        add_bos_token: If True, adds beginning-of-sequence token.
        add_eos_token: If True, adds end-of-sequence token after assistant messages.
        tokenizer: HuggingFace/Unsloth tokenizer. Required if tokenize=True.

    Returns:
        Formatted string with template applied, or list of token IDs if tokenize=True.
    """
    if template is None:
        template = get_alpaca_template()

    if role_mapping:
        messages = _apply_role_mapping(messages, role_mapping)

    parts = []
    last_was_user = False

    for i, msg in enumerate(messages):
        role = msg.get("role", msg.get("from", "user"))
        content = msg.get("content", msg.get("value", ""))

        is_last = i == len(messages) - 1
        last_was_user = role == "user"

        if role == "system" and template.system_template:
            parts.append(template.system_template.format(system=content))
        elif role == "user":
            parts.append(template.user_template.format(instruction=content, input=""))
        elif role == "assistant":
            assistant_text = template.assistant_template.format(output=content)
            # Add EOS token after assistant messages if configured
            if add_eos_token and template.eos_token:
                assistant_text += template.eos_token
            parts.append(assistant_text)

    # Handle add_generation_prompt - add assistant prefix if last message was from user
    if add_generation_prompt and last_was_user:
        # Get the assistant prefix from the template
        if template.assistant_template:
            # Extract the prefix part before {output}
            assistant_prefix = template.assistant_template.split("{output}")[0]
            if assistant_prefix:
                parts.append(assistant_prefix)

    result = template.separator.join(parts)

    if tokenize:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when tokenize=True")
        return tokenizer.encode(result, add_special_tokens=False)

    return result


def _apply_role_mapping(
    messages: list[dict],
    role_mapping: Dict[str, str],
) -> list[dict]:
    """Apply role mapping to messages."""
    mapped = []
    for msg in messages:
        role = msg.get("role", msg.get("from", "user"))
        mapped_role = role_mapping.get(role, role)
        # Ensure content/value is preserved in the mapped message
        content = msg.get("content", msg.get("value", ""))
        mapped.append({**msg, "role": mapped_role, "content": content})
    return mapped


def standardize_sharegpt(
    data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    role_mapping: Optional[Dict[str, str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Convert ShareGPT format to ChatML role/content format.

    ShareGPT format uses 'from' and 'value' keys:
        [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]

    ChatML format uses 'role' and 'content' keys (Hugging Face standard):
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]

    This function is essential when using ShareGPT-style datasets with models
    that expect ChatML format. Always call this before training on ShareGPT data.

    Args:
        data: List of conversations or list of messages. Can be:
            - Single conversation: [{"from": "human", "value": "Hi"}, ...]
            - Multiple conversations: [[{"from": "human", "value": "Hi"}, ...], ...]
        role_mapping: Mapping from ShareGPT roles to standard roles.
                    Default: {"human": "user", "gpt": "assistant"}
        column_mapping: Optional mapping for source column names.
                       Default: {"from": "from", "value": "value"}

    Returns:
        List of conversations in ChatML format with 'role' and 'content' keys.
    """
    if role_mapping is None:
        role_mapping = {"human": "user", "gpt": "assistant"}

    if column_mapping is None:
        column_mapping = {"from": "from", "value": "value"}

    # Try to use Unsloth's implementation first
    try:
        from unsloth import standardize_sharegpt as unsloth_standardize_sharegpt

        return unsloth_standardize_sharegpt(data)
    except ImportError:
        pass

    standardized = []

    for conversation in data:
        if isinstance(conversation, list):
            conv = []
            for msg in conversation:
                from_role = msg.get(column_mapping["from"], msg.get("role", "user"))
                content = msg.get(column_mapping["value"], msg.get("content", ""))
                role = role_mapping.get(from_role, from_role)
                conv.append({"role": role, "content": content})
            standardized.append(conv)
        else:
            standardized.append(conversation)

    return standardized


def to_sharegpt(
    data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    column_mapping: Optional[Dict[str, str]] = None,
    role_mapping: Optional[Dict[str, str]] = None,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Convert ChatML or other formats to ShareGPT format.

    This is the inverse of standardize_sharegpt. Useful when you need to
    convert datasets to ShareGPT format for compatibility with certain tools.

    Args:
        data: List of conversations in ChatML format
        column_mapping: Mapping for column names.
                       Default: {"role": "from", "content": "value"}
        role_mapping: Mapping from standard roles to ShareGPT roles.
                     Default: {"user": "human", "assistant": "gpt"}

    Returns:
        List of conversations in ShareGPT format with 'from' and 'value' keys.
    """
    if column_mapping is None:
        column_mapping = {"role": "from", "content": "value"}

    if role_mapping is None:
        role_mapping = {"user": "human", "assistant": "gpt"}

    sharegpt_data = []

    for conversation in data:
        if isinstance(conversation, list):
            conv = []
            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                from_role = role_mapping.get(role, role)
                conv.append({column_mapping["from"]: from_role, column_mapping["value"]: content})
            sharegpt_data.append(conv)
        else:
            sharegpt_data.append(conversation)

    return sharegpt_data


def merge_conversations(
    conversations: List[List[Dict[str, Any]]],
    extension_length: int = 3,
    seed: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Merge single-turn conversations into multi-turn conversations.

    This function extends datasets by merging consecutive conversations
    into multi-turn dialogues. Useful for creating multi-turn training data
    from single-turn datasets (conversation_extension in Unsloth).

    Args:
        conversations: List of conversations (each conversation is a list of messages)
        extension_length: Number of conversations to merge into one (default: 3)
        seed: Random seed for reproducibility

    Returns:
        List of merged multi-turn conversations
    """
    if seed is not None:
        random.seed(seed)

    if len(conversations) < extension_length:
        return conversations

    merged = []
    i = 0

    while i < len(conversations):
        num_to_merge = min(extension_length, len(conversations) - i)

        if num_to_merge < 1:
            break

        merged_conv = []
        for j in range(num_to_merge):
            merged_conv.extend(conversations[i + j])

        merged.append(merged_conv)
        i += num_to_merge

    return merged


def get_chat_template(
    tokenizer: Any,
    chat_template: str = "alpaca",
    mapping: Optional[Dict[str, str]] = None,
    map_eos_token: bool = True,
    map_bos_token: bool = True,
) -> Any:
    """Apply chat template to tokenizer.

    This function integrates with the tokenizer's Jinja chat template,
    following Unsloth's API for chat template configuration.

    Args:
        tokenizer: HuggingFace tokenizer object
        chat_template: Template name (e.g., "llama-3", "gemma-3", "phi-4", "qwen-2.5")
        mapping: Optional mapping for column names (e.g., {"role": "from", "content": "value"})
        map_eos_token: If True, maps EOS token (recommended: True)
        map_bos_token: If True, maps BOS token (recommended: True)

    Returns:
        Tokenizer with chat template applied
    """
    # Try to use Unsloth's native implementation first for maximum correctness
    try:
        from unsloth.chat_templates import get_chat_template as unsloth_get_chat_template

        return unsloth_get_chat_template(
            tokenizer=tokenizer,
            chat_template=chat_template,
            mapping=mapping,
            map_eos_token=map_eos_token,
            map_bos_token=map_bos_token,
        )
    except ImportError:
        pass

    # Fallback to local implementation
    template = get_template(chat_template)

    if hasattr(tokenizer, "apply_chat_template"):
        jinja_template = _convert_to_jinja(template)
        tokenizer.chat_template = jinja_template
    else:
        tokenizer.chat_template = template.user_template

    if mapping:
        tokenizer.role_mapping = mapping

    if map_eos_token and template.eos_token:
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                tokenizer.chat_template = tokenizer.chat_template.replace(
                    "{{ eos_token }}", tokenizer.eos_token
                )
        elif template.eos_token and hasattr(tokenizer, "eos_token"):
            tokenizer.eos_token = template.eos_token

    if map_bos_token and hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace(
                "{{ bos_token }}", tokenizer.bos_token
            )

    return tokenizer


def _convert_to_jinja(template: InstructionTemplate) -> str:
    """Convert AutoTrain template to Jinja chat template string.

    Uses proper Jinja2 template construction instead of fragile string replacement.
    """
    system_part = template.system_template.replace("{system}", "{{ add_generation_prompt }}")
    user_part = template.user_template.replace("{instruction}", "{{ message['content'] }}")
    assistant_part = template.assistant_template.replace("{output}", "{{ message['content'] }}")

    jinja_lines = [
        "{% for message in messages %}",
        "{% if message['role'] == 'system' %}",
        system_part,
        "{% elif message['role'] == 'user' %}",
        user_part,
        "{% elif message['role'] == 'assistant' %}",
        assistant_part,
        "{% endif %}",
        "{% endfor %}",
        "{% if add_generation_prompt %}",
        assistant_part.split("{{ message['content'] }}")[0]
        if "{{ message['content'] }}" in assistant_part
        else "",
        "{% endif %}",
    ]

    return "\n".join(jinja_lines)


__all__ = [
    "InstructionTemplate",
    "TemplateType",
    "get_template",
    "auto_detect_template",
    "create_custom_template",
    "format_sample",
    "format_samples_batch",
    "apply_chat_template",
    "standardize_sharegpt",
    "to_sharegpt",
    "merge_conversations",
    "get_chat_template",
    "CHAT_TEMPLATES",
    "TEMPLATE_REGISTRY",
]
