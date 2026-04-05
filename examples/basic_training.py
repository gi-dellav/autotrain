"""Basic training example with AutoTrain.

This example demonstrates the simplest way to fine-tune a model using AutoTrain.
It shows how to:
1. Initialize a model
2. Configure PEFT settings with dynamic parameters (_fn functions)
3. Add training samples
4. Train the model

Run with: python examples/basic_training.py
"""

from autotrain import Model, PEFTConfig

# Initialize the model with a pre-quantized model
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=2,
)

# Configure PEFT settings with dynamic parameters (_fn functions)
# These functions are called each training iteration to adjust values
peft_config = PEFTConfig(
    r=16,
    # lora_alpha_fn: Linear increase from 32 to 64 over iterations
    lora_alpha_fn=lambda i: 32 + i * 4,
    # lora_dropout_fn: Linear decrease from 0.1 to 0.0 over iterations
    lora_dropout_fn=lambda i: max(0.0, 0.1 - i * 0.02),
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    use_rslora=False,
    tuning_method="qlora",
)
model.set_peft_config(peft_config)

# Add training samples (instruction, input, output)
model.add_sample("What is 2+2?", "2+2=4")
model.add_sample("What is the capital of France?", "The capital of France is Paris.")
model.add_sample(
    "Explain photosynthesis.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
)

# Train the model
# k = number of synthetic samples to generate per input
# i = number of training iterations
model.train(k=100, i=10)

# Generate with the fine-tuned model
response = model.generate("What is 3+3?")
print(f"Response: {response}")
