"""Self-tuning for a specific task example.

This example demonstrates using AutoTrain's self-tuning pipeline for a specific task
(code review). The self-tuning loop iteratively improves the model using:
Producer -> Solver -> Splitter -> Checker

Run with: python examples/self_tuning_task.py
"""

from autotrain import Model

# Initialize model for code review task
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=3,
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
