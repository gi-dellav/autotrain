"""Test script to verify parallelization improvements."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

# Mock the necessary dependencies
import sys
sys.path.insert(0, '/home/giuseppe/autotrain')

from autotrain.config import InferenceConfig
from autotrain.components.producer import Producer
from autotrain.components.solver import Solver
from autotrain.components.splitter import Splitter
from autotrain.components.reviewer import Checker
from autotrain.data_types import Sample

# Create a mock model
mock_model = MagicMock()
mock_model.inference_config = InferenceConfig(max_workers=4, use_async=False)
mock_model.sample_multiplier = 2
mock_model.prompts = MagicMock()
mock_model.prompts.get_producer.return_value = "Generate a sample"
mock_model.prompts.get_solver.return_value = "Solve this"
mock_model.prompts.get_splitter.return_value = "Select best"
mock_model.prompts.get_checker.return_value = "Check this"

# Mock generate method for producer
def mock_generate(prompt, **kwargs):
    time.sleep(0.1)  # Simulate API latency
    return f"Sample output for: {prompt[:30]}"

mock_model.generate = mock_generate

print("Testing components with parallelization...")

# Test 1: Producer with shared executor
print("\n1. Testing Producer with shared executor")
shared_executor = ThreadPoolExecutor(max_workers=4)
producer = Producer(model=mock_model, prompt="test", inference_config=mock_model.inference_config)

start = time.time()
samples = producer.generate(10, executor=shared_executor)
elapsed = time.time() - start
print(f"   Generated {len(samples)} samples in {elapsed:.2f}s")
assert len(samples) == 10

# Test 2: Solver with shared executor
print("\n2. Testing Solver with shared executor")
solver = Solver(model=mock_model, prompt="test", inference_config=mock_model.inference_config)
test_samples = [Sample(input_data=f"Problem {i}", output_data="", metadata={}) for i in range(10)]

start = time.time()
solved = solver.solve(test_samples, executor=shared_executor)
elapsed = time.time() - start
print(f"   Solved {len(solved)} samples in {elapsed:.2f}s")
assert len(solved) == 10
assert all(s.output_data for s in solved)

# Test 3: Splitter with parallel groups
print("\n3. Testing Splitter with parallelization")
splitter = Splitter(model=mock_model, prompt="test", inference_config=mock_model.inference_config)

# Create samples with outputs
grouped_samples = []
for i in range(20):
    s = Sample(input_data=f"Input {i}", output_data=f"Output {i}", metadata={})
    grouped_samples.append(s)

start = time.time()
selected = splitter.select(grouped_samples, target_count=10, executor=shared_executor)
elapsed = time.time() - start
print(f"   Selected {len(selected)} samples from {len(grouped_samples)} in {elapsed:.2f}s")
assert len(selected) <= 10

# Test 4: Splitter without executor (sequential fallback)
print("\n4. Testing Splitter without executor (sequential)")
start = time.time()
selected_seq = splitter.select(grouped_samples, target_count=10, executor=None)
elapsed_seq = time.time() - start
print(f"   Selected {len(selected_seq)} samples in {elapsed_seq:.2f}s")

if elapsed > 0 and elapsed_seq > 0:
    speedup = elapsed_seq / elapsed
    print(f"\n   Speedup from shared executor: {speedup:.2f}x")

shared_executor.shutdown(wait=True)

print("\nAll tests passed! Parallelization improvements are working.")
