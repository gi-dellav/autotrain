"""Base Model class for AutoTrain."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Callable

from autotrain.checkpoints import CheckpointManager
from autotrain.config import (
    InferenceConfig,
    PEFTConfig,
    Prompts,
    ScalableTrainingConfig,
    TrainingConfig,
)
from autotrain.data_types import Sample
from autotrain.expert import Expert
from autotrain.tools import Tool, ToolCallingConfig

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.templates import InstructionTemplate


class BaseModel:
    """
    Base class for AutoTrain models.
    """

    def __init__(
        self,
        model_name: str,
        sample_multiplier: int = 2,
        inference_config: Optional[InferenceConfig] = None,
        prompts: Optional[Prompts] = None,
        enable_checker: bool = False,
        checker_rewrite_mode: bool = False,
        checkpoint_dir: Optional[str] = None,
        keep_best_checkpoint: bool = True,
        scalable_config: Optional[ScalableTrainingConfig] = None,
        thinking: bool = True,
    ):
        self.model_name = model_name
        self.sample_multiplier = sample_multiplier
        self.inference_config = inference_config or InferenceConfig()
        self.prompts = prompts or Prompts()
        self.enable_checker = enable_checker
        self.checker_rewrite_mode = checker_rewrite_mode
        self._scalable_config = scalable_config or ScalableTrainingConfig()
        self.thinking = thinking

        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir or "./checkpoints", keep_best=keep_best_checkpoint
        )

        self._benchmark: Optional["Benchmark"] = None
        self._peft_config = PEFTConfig()
        self._training_config = TrainingConfig()
        self._current_iteration = 0
        self._template: Optional["InstructionTemplate"] = None

        self._model = None
        self._tokenizer = None
        self._fast_model = None
        self._is_model_loaded = False
        self._base_model_name = model_name

        self._producer = None
        self._solver = None
        self._splitter = None
        self._checker = None

        self._samples: List[Any] = []
        self._training_data: List[Dict[str, Any]] = []
        self.expert_training_data: List[Dict[str, Any]] = []
        self.collect_expert_data = False
        self._benchmark_history: List[Dict[str, Any]] = []

        self._tools = ToolCallingConfig()
        self._experts: List[tuple] = []

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def fast_model(self):
        return self._fast_model

    @property
    def is_loaded(self) -> bool:
        return self._is_model_loaded

    def set_peft_config(self, config: PEFTConfig) -> None:
        self._peft_config = config

    def get_peft_config(self) -> PEFTConfig:
        return self._peft_config

    def set_training_config(self, config: TrainingConfig) -> None:
        self._training_config = config

    def get_training_config(self) -> TrainingConfig:
        return self._training_config

    def set_scalable_config(self, **kwargs) -> None:
        if not kwargs: return
        for k, v in kwargs.items():
            if hasattr(self._scalable_config, k):
                setattr(self._scalable_config, k, v)

    def get_scalable_config(self) -> ScalableTrainingConfig:
        return self._scalable_config

    def add_expert(self, expert: Any, weight: float = 1.0, production_weight: Optional[float] = None, check_weight: float = 0.0) -> None:
        p_weight = production_weight if production_weight is not None else weight
        self._experts.append((expert, p_weight))
        from autotrain.core.management import add_expert
        add_expert(self, expert, production_weight=p_weight, check_weight=check_weight)


    def remove_expert(self, expert: Any) -> None:
        self._experts = [e for e in self._experts if e[0] != expert]
        from autotrain.core.management import remove_expert
        remove_expert(self, expert)

    def clear_experts(self) -> None:
        self._experts = []
        if self._solver: self._solver.clear_experts()

    def get_experts(self) -> List[tuple]:
        return self._experts

    def set_benchmark(self, benchmark: "Benchmark") -> None:
        self._benchmark = benchmark

    def get_benchmark(self) -> Optional["Benchmark"]:
        return self._benchmark

    def save_checkpoint(self, iteration: int, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        from autotrain.core.management import save_checkpoint
        save_checkpoint(self, iteration, metadata)
        return f"checkpoint_{iteration}"

    def load_checkpoint(self, checkpoint_id: Optional[str] = None, iteration: Optional[int] = None) -> Any:
        from autotrain.core.management import load_checkpoint
        return load_checkpoint(self, checkpoint_id, iteration)

    def list_checkpoints(self) -> List[Any]:
        from autotrain.core.management import list_checkpoints
        return list_checkpoints(self)

    def export_expert_data(self, path: Union[str, Path]) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith(".jsonl"):
            with open(path, "w") as f:
                for item in self.expert_training_data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(path, "w") as f:
                json.dump(self.expert_training_data, f, indent=2)

    def add_tool(self, tool: Union[Tool, Callable[..., Any]], name: Optional[str] = None) -> Tool:
        return self._tools.add_tool(tool, name)

    def list_tools(self) -> List[str]:
        return self._tools.list_tools()

    def has_tool(self, name: str) -> bool:
        return name in self._tools.list_tools()

    def remove_tool(self, name: str) -> bool:
        return self._tools.remove_tool(name)

    def clear_tools(self) -> None:
        self._tools.clear_tools()

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get_tool(name)

    def create_tool(self, name: Optional[str] = None, description: Optional[str] = None):
        from autotrain.tools import create_tool
        
        orig_decorator = create_tool(name=name, description=description)
        
        def model_decorator(func):
            tool = orig_decorator(func)
            self.add_tool(tool)
            return tool
            
        return model_decorator

    def add_benchmark_sample(
        self, input_data: str, expected_output: str, mode: str = "exact_match"
    ) -> None:
        from autotrain.core.management import add_benchmark_sample
        add_benchmark_sample(self, input_data, expected_output, mode)

    def _init_components(self, experts: Optional[list] = None) -> None:
        from autotrain.core.training import _init_components
        _init_components(self, experts=experts)

    def add_sample(self, sample: Any) -> None:
        """Add a training sample."""
        self._samples.append(sample)
        
        # Determine how to format for training data
        if hasattr(sample, "to_dict"):
            self._training_data.append(sample.to_dict())
        elif isinstance(sample, dict):
            self._training_data.append(sample)
        else:
            # Fallback for standard Sample/VisionSample
            from autotrain.data_types import Sample, VisionSample
            if isinstance(sample, VisionSample):
                self._training_data.append({
                    "input_data": sample.input_data,
                    "output_data": sample.output_data,
                    "images": sample.images,
                    "messages": sample.to_conversation(),
                    "metadata": sample.metadata,
                })
            elif isinstance(sample, Sample):
                self._training_data.append({
                    "input": sample.input_data,
                    "output": sample.output_data,
                    "messages": sample.to_conversation(),
                    "metadata": sample.metadata,
                })
            else:
                # Generic fallback
                self._training_data.append({"data": str(sample)})

    def get_samples(self) -> List[Any]:
        return self._samples

    def clear_samples(self) -> None:
        self._samples = []
        self._training_data = []

    def for_inference(self) -> None:
        if self._fast_model is not None:
            try:
                # Use duck-typing to call for_inference if available
                if hasattr(self._fast_model, "for_inference"):
                    self._fast_model.for_inference()
                    print("Model prepared for inference (method)")
                else:
                    # Fallback to class methods if available
                    import unsloth
                    if hasattr(self, "processor") and getattr(self, "processor") is not None: # Vision
                        unsloth.FastVisionModel.for_inference(self._fast_model)
                        print("Model prepared for inference (Vision)")
                    else:
                        unsloth.FastLanguageModel.for_inference(self._fast_model)
                        print("Model prepared for inference (Language)")
            except Exception as e:
                print(f"Warning: Could not prepare for inference: {e}")


    # Abstract/Subclass methods
    def load_model(self, **kwargs):
        raise NotImplementedError

    def _unload_model(self):
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', loaded={self._is_model_loaded})"
