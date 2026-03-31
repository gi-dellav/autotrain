"""Basic training example with AutoTrain.

This example demonstrates the simplest way to fine-tune a model using AutoTrain.
It shows how to:
1. Initialize a model
2. Add training samples
3. Train the model

Run with: python examples/basic_training.py
"""

from autotrain import Model

# Initialize the model with a pre-quantized model
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=2,
)

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
