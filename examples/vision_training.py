"""Vision model training example.

This example demonstrates fine-tuning a Vision Language Model (VLM)
for image understanding tasks.

Run with: python examples/vision_training.py
"""

from autotrain import VisionModel, VisionSample
from pathlib import Path

# Initialize vision model
model = VisionModel(
    model_name="unsloth/Qwen3.5-27B-GGUF",
    sample_multiplier=2,
)

# Add vision training samples (image path + instruction + output)
# Note: Replace with actual image paths
samples = [
    VisionSample(
        image_path="path/to/cat_image.jpg",
        instruction="What do you see in this image?",
        output="I see a cat sitting on a couch.",
    ),
    VisionSample(
        image_path="path/to/food_image.jpg",
        instruction="Describe the food in this image.",
        output="The image shows a plate of pasta with tomato sauce.",
    ),
    VisionSample(
        image_path="path/to/code_image.jpg",
        instruction="What code is shown in this screenshot?",
        output="The code shows a Python function that calculates factorial.",
    ),
]

for sample in samples:
    model.add_sample(sample)

# Train the vision model
# k = synthetic samples per input
# i = training iterations
model.train(k=20, i=3)

# Generate with vision model
# Note: Requires actual image for inference
response = model.generate(prompt="Describe this image", image_path="path/to/test_image.jpg")
print(f"Vision model response: {response}")
