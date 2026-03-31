"""Checkpoint recovery example.

This example demonstrates automatic checkpointing and crash recovery
during training.

Run with: python examples/checkpoint_recovery.py
"""

from autotrain import Model

# Initialize model with checkpoint configuration
model = Model(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    checkpoint_dir="./model_checkpoints",
    keep_best_checkpoint=True,
)

# Add training samples
model.add_sample("What is machine learning?", "Machine learning is a subset of AI.")
model.add_sample("What is deep learning?", "Deep learning uses neural networks.")
model.add_sample("What is NLP?", "NLP stands for Natural Language Processing.")

# Train the model
# Checkpoints are automatically saved after each iteration
model.train(k=50, i=10)

# List available checkpoints
checkpoints = model.list_checkpoints()
print(f"\nAvailable checkpoints:")
for cp in checkpoints:
    print(f"  - Iteration {cp.get('iteration', '?')}: {cp.get('path', 'N/A')}")

# Restore the best checkpoint
model.restore_best_checkpoint()
print("\nRestored best checkpoint")

# Or restore a specific iteration
if checkpoints:
    specific_cp = checkpoints[0]
    model.load_checkpoint(checkpoint_id=specific_cp.get("id"))
    print(f"Loaded checkpoint: {specific_cp}")

# Generate with recovered model
response = model.generate("What is reinforcement learning?")
print(f"\nResponse from recovered model: {response}")
