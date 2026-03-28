# AutoTrain

Python framework for LLM self-tuning and distillation based on Unsloth and LiteLLM.

## Features

- **Iterative Self-Tuning**: Automatic training loop with Producer → Solver → Splitter → Reviewer pipeline
- **Multiple Experts**: Support for multiple teacher models with weighted production rates (e.g., 50% from Expert1, 25% from Expert2, 25% from student model)
- **Checkpoints**: Automatic checkpointing for crash recovery with best-model retention
- **Benchmarks**: Evaluate model quality during training with:
  - Exact match evaluation
  - LLM-as-judge evaluation
  - Early stopping based on benchmark stagnation
- **Unsloth API Access**: Full control over hyperparameters (LoRA config, training params)
- **Export Options**:
  - GGUF format for llama.cpp
  - Push to Hugging Face Hub
  - Ollama model creation
- **DPO Training**: Direct Preference Optimization for alignment with human preferences
- **Instruction Templates**: Pre-built templates for Alpaca, ChatML, Llama-3, Mistral, Gemma, and Phi
- **Logging & Observability**: TensorBoard and Weights & Biases integration
- **Scalable Training**:
  - Gradient checkpointing for memory efficiency
  - Mixed precision (fp16, bf16, fp32)
  - Automatic batch size tuning

## Installation

```bash
pip install autotrain
```

For development:
```bash
pip install autotrain[dev]
```

With logging support:
```bash
pip install autotrain[logging]
```

With DPO support:
```bash
pip install autotrain[dpo]
```

All features:
```bash
pip install autotrain[all]
```

## Quick Start

### Basic Training

```python
from autotrain import Model, Sample

# Create model
model = Model(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    sample_multiplier=2,
)

# Add initial samples
model.add_sample("What is 2+2?", "2+2=4")
model.add_sample("What is the capital of France?", "The capital of France is Paris.")

# Train
model.train(k=100, i=10)
```

### Multiple Experts with Weighted Production

```python
from autotrain import Model, Expert

# Create model
model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")

# Create multiple experts
expert1 = Expert(model_name="gpt-4", production_rate=0.5)
expert2 = Expert(model_name="claude-3-sonnet", production_rate=0.25)

# Train with weighted experts
# This means: 50% samples from expert1, 25% from expert2, 25% from model
model.train(
    k=100,
    i=10,
    experts=[(expert1, 0.5), (expert2, 0.25)]
)
```

### Benchmark Evaluation

```python
from autotrain import Model, Benchmark, EvaluationMode

# Create model
model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")

# Create benchmark
benchmark = Benchmark(
    name="math_benchmark",
    expert_model_name="gpt-4"  # For LLM judging
)

# Add benchmark samples (NOT used for training, only evaluation)
benchmark.add_sample(
    input_data="What is 15 × 23?",
    expected_output="345",
    evaluation_mode=EvaluationMode.EXACT_MATCH
)

benchmark.add_sample(
    input_data="Explain quantum entanglement",
    expected_output="Quantum entanglement is a phenomenon where particles become correlated...",
    evaluation_mode=EvaluationMode.LLM_JUDGE
)

# Train with benchmark evaluation and early stopping
model.train(
    k=100,
    i=50,
    benchmark=benchmark,
    early_stopping=True,
    early_stopping_patience=5,  # Stop if no improvement in 5 iterations
    early_stopping_threshold=0.01,
    keep_best_model=True,  # Keep the best model even if not last iteration
)

# Export benchmark results
benchmark.export_results("./benchmark_results.json")
```

### Checkpointing

```python
from autotrain import Model

model = Model(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    checkpoint_dir="./checkpoints",
    keep_best_checkpoint=True
)

# Train with automatic checkpointing
model.train(
    k=100,
    i=50,
    checkpoint_every=5,  # Save checkpoint every 5 iterations
)

# Resume from latest checkpoint
model.train(
    k=100,
    i=50,
    resume_from_checkpoint=True
)

# Restore best checkpoint (by benchmark accuracy)
model.restore_best_checkpoint()

# List checkpoints
checkpoints = model.list_checkpoints()
for ckpt in checkpoints:
    print(f"Iteration {ckpt.iteration}: {ckpt.checkpoint_id}")
```

### Hyperparameter Configuration

```python
from autotrain import Model

model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")

# Configure PEFT/LoRA parameters
model.set_peft_config(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Configure training hyperparameters
model.set_training_config(
    epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=10,
    weight_decay=0.01,
)

# Train
model.train(k=100, i=10)
```

### Export to GGUF

```python
from autotrain import Model

model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")
model.load_model()

# Train...
model.train(k=100, i=10)

# Export to GGUF
gguf_path = model.export_gguf(
    output_path="./exports/my_model",
    quantization="q4_k_m"  # Options: q4_k_m, q5_k_m, q8_0, f16, etc.
)
```

### Push to Hugging Face

```python
from autotrain import Model
import os

# Set HF_TOKEN environment variable or pass token parameter
os.environ["HF_TOKEN"] = "your_huggingface_token"

model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")
model.load_model()

# Train...
model.train(k=100, i=10)

# Push to Hugging Face
model.push_to_huggingface(
    repo_id="username/my-finetuned-model",
    private=False,
    merge_adapter=True  # Merge LoRA adapter into base model
)
```

### Export to Ollama

```python
from autotrain import Model

model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")
model.load_model()

# Train...
model.train(k=100, i=10)

# Export to Ollama
model.export_to_ollama(
    name="my-model",
    template="{{- if .System }}\n### System:\n{{ .System }}\n\n{{ end }}\n### User:\n{{ .Prompt }}\n\n### Response:\n{{ .Response }}",
    system_prompt="You are a helpful assistant."
)

# Then run in terminal:
# ollama create my-model -f Modelfile.my-model
```

### Expert Methods

```python
from autotrain import Expert

expert = Expert(
    model_name="gpt-4",
    api_key="your_api_key"
)

# Enable caching
expert.enable_cache()

# Solve problems
solution = expert.solve("What is the capital of France?")

# Review a sample
from autotrain import Sample
sample = Sample(input_data="2+2=?", output_data="5")
review = expert.review(sample)
print(f"Score: {review['score']}/10")

# Compare two outputs
comparison = expert.compare(
    input_data="Explain gravity",
    output_a="Gravity is a force...",
    output_b="Gravity is the curvature of spacetime..."
)
print(f"Winner: {comparison['winner']}")

# Rate an output
rating = expert.rate(
    input_data="Write a poem",
    output_data="Roses are red...",
    scale=10,
    criteria=["creativity", "rhyme", "meaning"]
)
```

### DPO Training

```python
from autotrain import Model
from autotrain.dpo import DPOTrainer, DPOConfig

# Create model
model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")

# Create DPO trainer
dpo_config = DPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    epochs=3,
    batch_size=4,
)
dpo_trainer = DPOTrainer(model, dpo_config)

# Add preference samples (prompt, chosen, rejected)
dpo_trainer.add_preference_sample(
    prompt="What is 2+2?",
    chosen="2+2=4. The sum of 2 and 2 is 4.",
    rejected="2+2=5",  # Incorrect answer
)
dpo_trainer.add_preference_sample(
    prompt="Explain gravity",
    chosen="Gravity is a fundamental force that attracts objects with mass toward each other.",
    rejected="Gravity is when things fall down.",  # Less complete
)

# Train with DPO
dpo_trainer.train(output_dir="./dpo_output")
```

### Instruction Templates

```python
from autotrain import Model
from autotrain.templates import (
    get_template,
    auto_detect_template,
    create_custom_template,
    format_sample,
)

# Auto-detect template based on model name
template = auto_detect_template("meta-llama/Llama-3-8b")
print(f"Using template: {template.name}")

# Or get a specific template
alpaca_template = get_template("alpaca")
chatml_template = get_template("chatml")
llama3_template = get_template("llama3")

# Format a sample
formatted = format_sample(
    instruction="What is the capital of France?",
    output="Paris is the capital of France.",
    model_name="llama-3",  # Auto-detects Llama-3 template
)

# Create custom template
custom_template = create_custom_template(
    name="my_template",
    system_template="<|system|>\n{system}",
    user_template="<|user|>\n{instruction}",
    assistant_template="<|assistant|>\n{output}",
)

# Use template in model
model = Model(model_name="unsloth/llama-3-8b-bnb-4bit")
model._template = llama3_template  # Set template for training
```

### Logging & Observability

```python
from autotrain import Model
from autotrain.logger import get_logger, AutoTrainLogger, LoggingConfig

# Create logger with TensorBoard
logger = get_logger(
    log_dir="./logs",
    enable_tensorboard=True,
    enable_wandb=False,
    run_name="my_training_run",
)

# Or create with WandB
wandb_logger = get_logger(
    log_dir="./logs",
    enable_tensorboard=True,
    enable_wandb=True,
    wandb_project="my_autotrain_project",
    wandb_entity="my_username",
)

# Log metrics
logger.log_scalar("loss", 0.5, step=1)
logger.log_metrics({"accuracy": 0.9, "f1": 0.85}, step=1)

# Log benchmark results
logger.log_benchmark_results(
    benchmark_name="math_benchmark",
    accuracy=0.85,
    average_score=0.82,
    total_samples=100,
    iteration=5,
)

# Log training progress
logger.log_training_progress(
    iteration=10,
    loss=0.45,
    samples_count=500,
    learning_rate=1e-4,
)

# Close logger when done
logger.close()

# View TensorBoard
# tensorboard --logdir ./logs/tensorboard/my_training_run
```

### Scalable Training

```python
from autotrain import Model
from autotrain.model import ScalableTrainingConfig

# Configure scalable training
scalable_config = ScalableTrainingConfig(
    gradient_checkpointing=True,      # Save memory with gradient checkpointing
    mixed_precision="fp16",           # Use fp16 mixed precision (options: fp16, bf16, fp32)
    batch_size_auto_tune=False,       # Auto-find optimal batch size
    max_memory_mb=16384,              # Max GPU memory to use (for auto-tuning)
    num_workers=4,                    # DataLoader workers
    pin_memory=True,                  # Pin memory for faster CPU->GPU transfer
)

# Create model with scalable config
model = Model(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    scalable_config=scalable_config,
)

# Or update config later
model.set_scalable_config(
    gradient_checkpointing=True,
    mixed_precision="bf16",
    batch_size_auto_tune=True,  # Auto-tune batch size based on available memory
)

# Auto-tune batch size manually
optimal_batch_size = model.auto_tune_batch_size()
print(f"Optimal batch size: {optimal_batch_size}")

# Train with scalable config
model.train(k=100, i=10)
```

## API Reference

### Model

| Method | Description |
|--------|-------------|
| `__init__(model_name, sample_multiplier, inference_config, prompts, enable_checker, checkpoint_dir, scalable_config)` | Initialize model |
| `load_model(max_seq_length, dtype, load_in_4bit)` | Load the unsloth model |
| `train(k, i, experts, initial_samples, benchmark, early_stopping, checkpoint_every, keep_best_model, resume_from_checkpoint)` | Train the model |
| `set_peft_config(r, lora_alpha, lora_dropout, bias, target_modules)` | Configure LoRA parameters |
| `set_training_config(epochs, batch_size, learning_rate, ...)` | Configure training hyperparameters |
| `set_scalable_config(gradient_checkpointing, mixed_precision, batch_size_auto_tune, ...)` | Configure scalable training |
| `auto_tune_batch_size()` | Auto-find optimal batch size |
| `add_expert(expert, production_weight, review_weight, check_weight)` | Add an expert with weights |
| `set_benchmark(benchmark, name, expert_model_name)` | Set benchmark for evaluation |
| `save_checkpoint(iteration, metadata)` | Save a checkpoint |
| `load_checkpoint(checkpoint_id, iteration)` | Load a checkpoint |
| `restore_best_checkpoint()` | Restore best checkpoint |
| `export_gguf(output_path, quantization, merge_adapter)` | Export to GGUF format |
| `push_to_huggingface(repo_id, token, private, merge_adapter)` | Push to HF Hub |
| `export_to_ollama(name, gguf_path, template, system_prompt)` | Export to Ollama |
| `generate(prompt, temperature, max_tokens, top_p)` | Generate text |

### DPOTrainer

| Method | Description |
|--------|-------------|
| `__init__(model, dpo_config, inference_config)` | Initialize DPO trainer |
| `add_preference_sample(prompt, chosen, rejected, metadata)` | Add a preference sample |
| `add_preference_samples(samples, metadata_list)` | Add multiple preference samples |
| `generate_preference_samples(expert, prompts, count_per_prompt)` | Generate samples with expert |
| `train(eval_samples, output_dir)` | Train with DPO |
| `export_dataset(path, format)` | Export preference dataset |
| `import_dataset(path, format)` | Import preference dataset |
| `get_statistics()` | Get dataset statistics |

### Instruction Templates

| Function | Description |
|----------|-------------|
| `get_template(name)` | Get a pre-defined template (alpaca, chatml, llama3, mistral, gemma, phi) |
| `auto_detect_template(model_name)` | Auto-detect template from model name |
| `create_custom_template(name, system_template, user_template, ...)` | Create custom template |
| `format_sample(instruction, output, model_name, template)` | Format a single sample |
| `format_samples_batch(samples, template, model_name)` | Format batch of samples |
| `apply_chat_template(messages, template)` | Apply chat template to messages |

### Logger

| Method | Description |
|--------|-------------|
| `__init__(name, config, run_name)` | Initialize logger |
| `log_scalar(name, value, step)` | Log a scalar value |
| `log_metrics(metrics, step)` | Log multiple metrics |
| `log_text(name, text, step)` | Log text data |
| `log_benchmark_results(benchmark_name, accuracy, ...)` | Log benchmark results |
| `log_training_progress(iteration, loss, samples_count, ...)` | Log training progress |
| `log_system_info(info)` | Log system information |
| `log_completion(summary)` | Log training completion |
| `close()` | Close logging backends |

### ScalableTrainingConfig

| Property | Description |
|----------|-------------|
| `gradient_checkpointing` | Enable gradient checkpointing for memory efficiency |
| `mixed_precision` | Mixed precision mode: "fp16", "bf16", "fp32" |
| `batch_size_auto_tune` | Auto-find optimal batch size |
| `max_memory_mb` | Maximum GPU memory to use for auto-tuning |
| `num_workers` | Number of DataLoader workers |
| `pin_memory` | Pin memory for faster CPU->GPU transfer |

### Expert

| Method | Description |
|--------|-------------|
| `__init__(model_name, production_rate, inference_config, prompts, api_key, api_base)` | Initialize expert |
| `produce(task, count)` | Generate training samples |
| `solve(input_data, system_prompt, temperature)` | Solve a problem |
| `select(samples)` | Select best sample from options |
| `review(sample, criteria)` | Review a sample |
| `check(input_data, output_data, strict)` | Verify correctness |
| `compare(input_data, output_a, output_b)` | Compare two outputs |
| `rate(input_data, output_data, scale, criteria)` | Rate an output |
| `enable_cache()` / `disable_cache()` / `clear_cache()` | Cache management |
| `set_production_rate(rate)` | Set production rate |
| `set_inference_config(temperature, max_tokens, ...)` | Update inference config |

### Benchmark

| Method | Description |
|--------|-------------|
| `__init__(name, expert_model_name, expert_api_key)` | Initialize benchmark |
| `add_sample(input_data, expected_output, evaluation_mode, metadata)` | Add a benchmark sample |
| `evaluate(model, iteration, inference_config)` | Evaluate model on benchmark |
| `has_improved(threshold, iterations_to_check)` | Check if improved |
| `has_stagnated(threshold, iterations)` | Check if stagnated |
| `export(path, format)` | Export samples |
| `export_results(path)` | Export results history |
| `load(path, format)` | Load benchmark from file |

### CheckpointManager

| Method | Description |
|--------|-------------|
| `__init__(checkpoint_dir, max_checkpoints, keep_best)` | Initialize manager |
| `save(model, iteration, benchmark_accuracy, metadata)` | Save checkpoint |
| `load(model, checkpoint_id, iteration)` | Load checkpoint |
| `restore_best(model)` | Restore best checkpoint |
| `list_checkpoints()` | List all checkpoints |
| `get_best_checkpoint()` | Get best checkpoint info |
| `export_checkpoint(checkpoint_id, output_path)` | Export checkpoint |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Producer │───▶│  Solver  │───▶│ Splitter │───▶│Reviewer│ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │              │                                      │
│       │              ▼                                      │
│       │        ┌──────────┐                                 │
│       │        │ Experts  │ (Multiple, weighted)            │
│       │        │ - GPT-4  │ 0.5                             │
│       │        │ - Claude │ 0.25                            │
│       │        └──────────┘                                 │
│       ▼                                                     │
│  ┌──────────┐                                               │
│  │ Checkpoint│ ◀─────── Benchmark Evaluation               │
│  │  Manager │        - Exact Match                         │
│  └──────────┘        - LLM Judge                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## License

Apache 2.0
