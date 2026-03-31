# AutoTrain

Python framework for LLM self-tuning and distillation based on Unsloth and LiteLLM.

## Features

- **Iterative Self-Tuning**: Producer → Solver → Splitter → Reviewer training pipeline
- **Multiple Experts**: Weighted teacher models (GPT-4, Claude, etc.) for diverse knowledge
- **Automatic Checkpointing**: Crash recovery with best-model retention
- **Benchmark Evaluation**: Exact match & LLM-as-judge with early stopping
- **Flexible Export**: GGUF, Hugging Face Hub, Ollama formats
- **DPO Training**: Direct Preference Optimization for alignment
- **Instruction Templates**: Alpaca, ChatML, Llama-3, Mistral, Gemma, Phi
- **Observability**: TensorBoard & Weights & Biases integration
- **Scalable Training**: Gradient checkpointing, mixed precision, auto batch tuning

## Installation

```bash
pip install autotrain
```

### Optional Dependencies
```bash
# Development tools
pip install autotrain[dev]

# Logging support
pip install autotrain[logging]

# DPO support
pip install autotrain[dpo]

# All features
pip install autotrain[all]
```

## Quick Start

```python
from autotrain import Model

# Initialize model
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=2
)

# Add training samples
model.add_sample("What is 2+2?", "2+2=4")
model.add_sample("What is the capital of France?", "The capital of France is Paris.")

# Train model
model.train(k=100, i=10)
```

## Advanced Usage

### Multiple Experts
```python
from autotrain import Model, Expert

model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")
expert1 = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", production_rate=0.5)
expert2 = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", production_rate=0.25)

model.train(k=100, i=10, experts=[expert1, expert2])
```

### Export Models
```python
# Export to GGUF
model.export_gguf("./model", quantization="q4_k_m")

# Push to Hugging Face
model.push_to_huggingface("username/model-name")

# Export to Ollama
model.export_to_ollama("my-model")
```

## Documentation

For detailed API reference and advanced examples, see the [documentation](docs/).

## License

Apache 2.0