"""DPO (Direct Preference Optimization) training example.

This example demonstrates how to use DPO to align models with human preferences
using chosen/rejected response pairs.

Run with: python examples/dpo_training.py
"""

from autotrain import Model
from autotrain.dpo import DPOTrainer, DPOConfig, PreferenceSample

# Initialize model
model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")

# Configure DPO training
dpo_config = DPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    epochs=1,
    batch_size=2,
    learning_rate=5e-7,
)

# Create DPO trainer
trainer = DPOTrainer(model, dpo_config)

# Add preference samples (prompt, chosen, rejected)
preference_samples = [
    ("What is 2+2?", "2+2 equals 4.", "2+2 equals 5."),
    (
        "Explain gravity.",
        "Gravity is a force that attracts objects toward each other.",
        "Gravity is fake.",
    ),
    ("What is Python?", "Python is a high-level programming language.", "Python is a snake."),
    (
        "Explain photosynthesis.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Photosynthesis is when plants sleep.",
    ),
    (
        "What is AI?",
        "AI is artificial intelligence - machines that can perform tasks requiring human intelligence.",
        "AI is magic.",
    ),
]

for prompt, chosen, rejected in preference_samples:
    trainer.add_preference_sample(prompt, chosen, rejected)

# Train with DPO
print(f"Training with {trainer.get_statistics()['total_samples']} preference samples...")
result = trainer.train()

print(f"DPO training complete: {result}")

# Export the DPO dataset for inspection
trainer.export_dataset("./dpo_dataset.json")
