"""AutoTrain - An automated training library based on unsloth and litellm."""

__version__ = "1.0.0"

# Core model and training
# Benchmark
from autotrain.benchmark import (
    Benchmark,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSample,
    EvaluationMode,
)

# Checkpoint
from autotrain.checkpoints import CheckpointInfo, CheckpointManager

# Components
from autotrain.components import (
    Checker,
    ComponentConfig,
    ExpertWeight,
    Producer,
    Solver,
    Splitter,
)

# Vision components
from autotrain.components.vision_collator import VisionDataCollator
from autotrain.components.vision_producer import VisionProducer

# Configuration classes
from autotrain.config import (
    InferenceConfig,
    PEFTConfig,
    Prompts,
    ScalableTrainingConfig,
    TrainingConfig,
)
from autotrain.core import Model, train
from autotrain.core.vision_model import VisionModel
from autotrain.core.training import train_vision_model

# Data types
from autotrain.data_types import Sample, VisionSample

# Dataset
from autotrain.datasets import (
    AutoTrainDataset,
    DatasetConfig,
    DatasetFormatter,
)

# Expert
from autotrain.expert import Expert, ExpertPrompts, VisionExpert

# Templates
from autotrain.templates import (
    InstructionTemplate,
    TemplateType,
    apply_chat_template,
    auto_detect_template,
    create_custom_template,
    format_sample,
    format_samples_batch,
    get_template,
)

# Tools
from autotrain.tools import (
    Tool,
    ToolCallingConfig,
    create_tool,
    get_tool_schema,
    python,
    web_search,
    DEFAULT_TOOLS,
)

__all__ = [
    # Main classes
    "Model",
    "VisionModel",
    "Expert",
    "VisionExpert",
    # Training
    "train",
    "train_vision_model",
    # Configuration
    "InferenceConfig",
    "Prompts",
    "ExpertPrompts",
    "ComponentConfig",
    "PEFTConfig",
    "TrainingConfig",
    "ScalableTrainingConfig",
    "ExpertWeight",
    "DatasetConfig",
    # Components
    "Producer",
    "VisionProducer",
    "Solver",
    "Splitter",
    "Checker",
    "VisionDataCollator",
    # Benchmark
    "Benchmark",
    "EvaluationMode",
    "BenchmarkSample",
    "BenchmarkResult",
    "BenchmarkMetrics",
    # Checkpoint
    "CheckpointManager",
    "CheckpointInfo",
    # Dataset
    "AutoTrainDataset",
    "DatasetFormatter",
    # Templates
    "InstructionTemplate",
    "TemplateType",
    "get_template",
    "auto_detect_template",
    "create_custom_template",
    "format_sample",
    "format_samples_batch",
    "apply_chat_template",
    # Data classes
    "Sample",
    "VisionSample",
    # Tools
    "Tool",
    "ToolCallingConfig",
    "create_tool",
    "get_tool_schema",
    "python",
    "web_search",
    "DEFAULT_TOOLS",
]
