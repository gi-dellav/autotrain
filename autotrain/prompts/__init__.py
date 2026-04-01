"""Prompt system for AutoTrain.

This module provides two types of prompts:

1. **Baking Functions**: For components like Producer and Solver that need
   dynamic prompts based on specific topics or context.
   - `bake_producer(topic, format)` - Generate training samples
   - `bake_solver(topic, context)` - Solve problems

2. **Default Constants**: For components like Splitter and Checker
   that use fixed, detailed prompts.
   - `SPLITTER_DEFAULT` - Selecting best training samples
   - `CHECKER_DEFAULT` - Verifying solution correctness
"""

from .constants import CHECKER_DEFAULT, SPLITTER_DEFAULT
from .templates import bake_producer, bake_solver

__all__ = [
    "bake_producer",
    "bake_solver",
    "SPLITTER_DEFAULT",
    "CHECKER_DEFAULT",
]
