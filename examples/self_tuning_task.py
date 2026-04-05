"""Self-tuning for a specific task example.

This example demonstrates using AutoTrain's self-tuning pipeline for a specific task
(code review) with custom prompts. The self-tuning loop iteratively improves the model
using: Producer -> Solver -> Splitter -> Checker

This example also shows how to configure custom prompts for each component.

Run with: python examples/self_tuning_task.py
"""

from autotrain import Model, Prompts, InferenceConfig

# Custom prompts for each component
# These override the default prompts used in the self-tuning loop
custom_prompts = Prompts(
    # Producer prompt: instructs how to generate synthetic training data
    producer="Generate a diverse code review training sample. "
    "Include: code snippet, potential issues, severity level. "
    "Format: JSON with 'code', 'review', 'severity' fields.",
    # Solver prompt: instructs how to solve/process the input
    solver="You are an expert code reviewer. Provide a detailed, constructive review "
    "of the following code. Focus on: correctness, performance, security, best practices.",
    # Splitter prompt: instructs how to split/select the best samples
    splitter="Select the highest quality code review based on: accuracy, detail, "
    "actionability, and clarity. Prefer reviews that identify real issues.",
    # Checker prompt: instructs how to verify correctness (if enabled)
    checker="Verify if the code review is accurate and helpful. Check: "
    "1) Correct identification of issues, 2) Reasonable severity assessment, "
    "3) Actionable suggestions.",
)

# Configure inference with dynamic temperature using _fn
# Temperature increases linearly from 0.3 to 0.7 over iterations
inference_config = InferenceConfig(
    temperature=0.4,
    temperature_fn=lambda i: 0.3 + i * 0.05,
    producer_temperature=1.0,
    splitter_temperature=0.3,
)

# Initialize model for code review task with custom prompts and inference config
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=3,
    prompts=custom_prompts,
    inference_config=inference_config,
)

# Add task-specific training samples
training_samples = [
    (
        "Review this Python code:\ndef add(a, b):\n    return a + b",
        "This is a simple function that adds two numbers. Good practices: clear naming, docstring could be added.",
    ),
    (
        "Review this code:\ndef divide(a, b):\n    return a / b",
        "Missing zero-division check. Should validate b != 0 before dividing.",
    ),
    (
        "Review this code:\nfor i in range(1000000):\n    print(i)",
        "Inefficient - prints to console in a loop. Consider using list comprehension or generator for large iterations.",
    ),
]

for instruction, output in training_samples:
    model.add_sample(instruction, output)

# Self-tuning: uses built-in LLM components to iteratively improve
# k = synthetic samples per input
# i = training iterations
model.train(k=50, i=5)

# Test the fine-tuned model
test_code = "Review this: x = [1,2,3]; print(x[10])"
response = model.generate(test_code)
print(f"Code review: {response}")
