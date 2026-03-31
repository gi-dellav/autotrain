"""Export model to GGUF format example.

This example demonstrates exporting a fine-tuned model to GGUF format
for use with llama.cpp, LM Studio, or other GGUF-compatible runners.

Run with: python examples/export_gguf.py
"""

from autotrain import Model

# Initialize and train model
model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")

model.add_sample("What is Python?", "Python is a programming language.")
model.add_sample("What is JavaScript?", "JavaScript is a web programming language.")

model.train(k=30, i=3)

# Export to GGUF format with quantization
# Options: q4_k_m, q5_k_m, q8_0, f16, f32
model.export_gguf(
    output_path="./exported_model",
    quantization="q4_k_m",
    merge_adapter=True,
)

print("Model exported to GGUF format!")

# Also supports pushing to Hugging Face Hub
# model.push_to_huggingface("username/my-model", token="hf_...")

# Or export to Ollama format
# model.export_to_ollama("./ollama_model", model_name="my-model")
