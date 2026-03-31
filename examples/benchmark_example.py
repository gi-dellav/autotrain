"""Benchmark evaluation example.

This example demonstrates how to evaluate model quality during training
using exact match, LLM-as-judge, or other evaluation modes.

Run with: python examples/benchmark_example.py
"""

from autotrain import Model
from autotrain.benchmark import (
    Benchmark,
    BenchmarkSample,
    EvaluationMode,
)

# Initialize model
model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")

# Add training samples
model.add_sample("What is 2+2?", "4")
model.add_sample("What is the capital of France?", "Paris")
model.add_sample("What is 5 * 6?", "30")

# Create benchmark for evaluation
benchmark = Benchmark(
    name="math_qa",
    expert_model_name="unsloth/Qwen3.5-27B-GGUF",
)

# Add benchmark samples with different evaluation modes
benchmark.add_sample(
    input_data="What is 2+2?",
    expected_output="4",
    evaluation_mode=EvaluationMode.EXACT_MATCH,
)

benchmark.add_sample(
    input_data="What is the capital of Italy?",
    expected_output="Rome",
    evaluation_mode=EvaluationMode.EXACT_MATCH,
)

benchmark.add_sample(
    input_data="Explain why the sky is blue.",
    expected_output="The sky is blue due to Rayleigh scattering of sunlight.",
    evaluation_mode=EvaluationMode.LLM_JUDGE,
)

# Set benchmark on model
model.set_benchmark(benchmark)

# Train with benchmark evaluation after each iteration
model.train(k=50, i=5)

# Evaluate model on benchmark
metrics = benchmark.evaluate(model, iteration=0)

print(f"\nBenchmark Results:")
print(f"  Accuracy: {metrics.exact_match_accuracy}")
print(f"  LLM Judge Score: {metrics.llm_judge_score}")
print(f"  Best Iteration: {benchmark.best_iteration}")

# List all benchmark results
print(f"\nBenchmark History:")
for i, result in enumerate(benchmark.results_history):
    print(f"  Iteration {i}: accuracy={result.exact_match_accuracy:.2f}")
