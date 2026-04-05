"""Benchmark evaluation example.

This example demonstrates how to evaluate model quality during training
using exact match, LLM-as-judge, or other evaluation modes.

This example also shows how to configure dynamic training parameters
using _fn functions (learning_rate_fn, epochs_fn, etc.).

Run with: python examples/benchmark_example.py
"""

from autotrain import Model, TrainingConfig, PEFTConfig, Benchmark, BenchmarkSample, EvaluationMode

# Configure training with dynamic parameters using _fn functions
# These adjust during training iterations for better convergence
training_config = TrainingConfig(
    epochs=3,
    # epochs_fn: Increase epochs as training progresses
    epochs_fn=lambda i: 3 + min(i, 5),
    batch_size=2,
    # learning_rate_fn: Learning rate warmup then decay
    learning_rate_fn=lambda i: 2e-4 * (1.0 if i < 2 else 0.5 ** (i - 1)),
    weight_decay=0.01,
    warmup_ratio=0.15,
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=100,
    seed=3407,
    scheduler_type="cosine",
)

# Initialize model
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
)
model.set_training_config(training_config)

# Configure PEFT with dynamic lora_rank_fn (linear increase each iteration)
peft_config = PEFTConfig(
    r=16,
    lora_alpha=32,
    # lora_rank_fn: Linear increase from 16 to 32 over 4 iterations
    lora_rank_fn=lambda i: 16 + i * 4,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"],
)
model.set_peft_config(peft_config)

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
