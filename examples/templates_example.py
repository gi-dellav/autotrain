"""Templates example.

This example demonstrates how to use and customize instruction templates
for different model architectures.

Run with: python examples/templates_example.py
"""

from autotrain import Model
from autotrain.templates import (
    get_template,
    auto_detect_template,
    create_custom_template,
    format_sample,
    apply_chat_template,
)

# Auto-detect template based on model name
template = auto_detect_template("unsloth/Qwen3.5-27B-GGUF")
print(f"Auto-detected template: {template.name}")

# Use specific templates
llama3_template = get_template("llama-3")
chatml_template = get_template("chatml")
alpaca_template = get_template("alpaca")

# Format a training sample with a specific template
formatted = format_sample(
    instruction="What is Python?",
    output="Python is a high-level programming language.",
    template=llama3_template,
)
print(f"\nFormatted with Llama-3 template:\n{formatted[:200]}...")

# Apply chat template to messages
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help?"},
    {"role": "user", "content": "What is 2+2?"},
]

chat_formatted = apply_chat_template(
    messages,
    template=chatml_template,
    add_generation_prompt=True,
)
print(f"\nChat formatted:\n{chat_formatted}")

# Create custom template
custom = create_custom_template(
    name="my_template",
    system_template="<|system|>\n{system}",
    user_template="<|user|>\n{instruction}",
    assistant_template="<|assistant|>\n{output}",
    eos_token="<|endoftext|>",
)
print(f"\nCustom template created: {custom.name}")

# Use custom template with model
model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")
model.set_template(custom)
print(f"\nModel template set to: {model.get_template().name}")
