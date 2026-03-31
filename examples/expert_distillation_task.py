"""Expert distillation example.

This example demonstrates using multiple expert models (GPT-4, Claude) to
generate high-quality training data through knowledge distillation.

Run with: python examples/expert_distillation_task.py
"""

from autotrain import Model, Expert

# Initialize base model to fine-tune
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=2,
)

# Create expert teachers
expert_gpt4 = Expert(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    production_rate=0.6,
)

expert_claude = Expert(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    production_rate=0.4,
)

# Add training samples
model.add_sample(
    "Explain quantum computing in simple terms.",
    "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, "
    "enabling parallel processing of many possibilities at once.",
)

model.add_sample(
    "What are the key differences between SQL and NoSQL databases?",
    "SQL databases are relational, use structured tables with schemas, and support complex queries. "
    "NoSQL databases are non-relational, use flexible schemas, and are better for unstructured data.",
)

# Train with expert distillation
# Experts generate high-quality responses that the model learns from
model.train(k=100, i=10, experts=[expert_gpt4, expert_claude])

# Generate with distilled model
response = model.generate("Explain machine learning")
print(f"Response: {response}")
