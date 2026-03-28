"""Tests for Model class."""

from unittest.mock import MagicMock

from autotrain import (
    InferenceConfig,
    Model,
    PEFTConfig,
    Prompts,
    Sample,
    TrainingConfig,
)

from autotrain.benchmark import Benchmark, EvaluationMode
from autotrain.expert import Expert


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_values(self):
        """Test default inference config values."""
        config = InferenceConfig()

        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0

    def test_custom_values(self):
        """Test custom inference config values."""
        config = InferenceConfig(
            temperature=0.9,
            max_tokens=2048,
            top_p=0.95,
        )

        assert config.temperature == 0.9
        assert config.max_tokens == 2048
        assert config.top_p == 0.95


class TestPrompts:
    """Tests for Prompts dataclass."""

    def test_default_prompts(self):
        """Test default prompt values."""
        prompts = Prompts()

        # Check that default values are None (new system uses getters)
        assert prompts.producer is None
        assert prompts.solver is None
        assert prompts.splitter is None
        assert prompts.reviewer is None
        assert prompts.checker is None

        # Check getter methods return detailed prompts
        producer_prompt = prompts.get_producer("test topic")
        assert "You are an expert data generator" in producer_prompt
        assert "test topic" in producer_prompt

        solver_prompt = prompts.get_solver()
        assert "You are an expert problem solver" in solver_prompt

        splitter_prompt = prompts.get_splitter()
        assert "You are an expert at selecting" in splitter_prompt

        reviewer_prompt = prompts.get_reviewer()
        assert "You are an expert reviewer" in reviewer_prompt

        checker_prompt = prompts.get_checker()
        assert "You are an expert verifier" in checker_prompt

    def test_custom_prompts(self):
        """Test custom prompt values."""
        prompts = Prompts(
            producer="Custom producer prompt",
            solver="Custom solver prompt",
        )

        assert prompts.producer == "Custom producer prompt"
        assert prompts.solver == "Custom solver prompt"
        assert prompts.splitter is None
        assert prompts.reviewer is None
        assert prompts.checker is None

        # Custom prompts should be returned by getters
        assert prompts.get_producer("test topic") == "Custom producer prompt"
        assert prompts.get_solver() == "Custom solver prompt"
        # Default constants for others
        assert "You are an expert at selecting" in prompts.get_splitter()


class TestSample:
    """Tests for Sample dataclass."""

    def test_sample_creation(self):
        """Test Sample creation."""
        sample = Sample(input_data="What is 2+2?", output_data="4")

        assert sample.input_data == "What is 2+2?"
        assert sample.output_data == "4"
        assert sample.metadata == {}

    def test_sample_with_metadata(self):
        """Test Sample creation with metadata."""
        sample = Sample(
            input_data="What is 2+2?", output_data="4", metadata={"source": "expert", "score": 10}
        )

        assert sample.metadata["source"] == "expert"
        assert sample.metadata["score"] == 10


class TestPEFTConfig:
    """Tests for PEFTConfig dataclass."""

    def test_default_values(self):
        """Test default PEFT config values."""
        config = PEFTConfig()

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.0
        assert config.bias == "none"
        assert config.target_modules is None

    def test_get_target_modules_llama(self):
        """Test getting target modules for Llama model."""
        config = PEFTConfig()
        modules = config.get_target_modules(model_type="llama")

        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "v_proj" in modules
        assert "o_proj" in modules

    def test_get_target_modules_custom(self):
        """Test getting target modules with custom setting."""
        config = PEFTConfig(target_modules=["custom_proj"])
        modules = config.get_target_modules(model_type="llama")

        assert modules == ["custom_proj"]


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default training config values."""
        config = TrainingConfig()

        assert config.epochs == 3
        assert config.batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-4
        assert config.seed == 3407


class TestModel:
    """Tests for Model class."""

    def test_model_init_default(self):
        """Test Model initialization with defaults."""
        model = Model()

        assert model.model_name == "unsloth/Qwen3.5-27B-GGUF"
        assert model.sample_multiplier == 2
        assert model.enable_checker is False
        assert model._is_model_loaded is False
        assert model._samples == []
        assert model._training_data == []

    def test_model_init_custom(self):
        """Test Model initialization with custom params."""
        config = InferenceConfig(temperature=0.9)
        prompts = Prompts(producer="Custom prompt")

        model = Model(
            model_name="unsloth/mistral-7b-bnb-4bit",
            sample_multiplier=3,
            inference_config=config,
            prompts=prompts,
            enable_checker=True,
            checkpoint_dir="./test_checkpoints",
        )

        assert model.model_name == "unsloth/mistral-7b-bnb-4bit"
        assert model.sample_multiplier == 3
        assert model.enable_checker is True
        assert model.inference_config.temperature == 0.9
        assert model.prompts.producer == "Custom prompt"

    def test_model_not_loaded(self):
        """Test model is not loaded by default."""
        model = Model()

        assert model.is_loaded is False
        assert model.model is None
        assert model.tokenizer is None
        assert model.fast_model is None

    def test_model_load(self, mock_unsloth):
        """Test model loading."""
        model = Model()
        model.load_model()

        assert model.is_loaded is True
        assert mock_unsloth.FastLanguageModel.from_pretrained.called

    def test_set_peft_config(self):
        """Test setting PEFT configuration."""
        from autotrain.config import PEFTConfig

        model = Model()
        config = PEFTConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model.set_peft_config(config)

        config = model.get_peft_config()
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_set_training_config(self):
        """Test setting training configuration."""
        from autotrain.config import TrainingConfig

        model = Model()
        config = TrainingConfig(
            epochs=3,
            batch_size=8,
            learning_rate=1e-4,
            warmup_steps=20,
        )
        model.set_training_config(config)

        config = model.get_training_config()
        assert config.epochs == 3
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.warmup_steps == 20

    def test_add_sample(self):
        """Test adding samples to model."""
        model = Model()
        sample = Sample(input_data="test input", output_data="test output")

        # Access internal list for testing
        model._samples.append(sample)

        assert len(model._samples) == 1
        assert model._samples[0].input_data == "test input"

    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        model = Model()
        samples = [Sample(input_data=f"input_{i}", output_data=f"output_{i}") for i in range(5)]

        model._samples.extend(samples)

        assert len(model._samples) == 5

    def test_add_expert(self, mock_litellm):
        """Test adding expert to model."""
        model = Model()
        expert = Expert(model_name="gpt-4", api_key="test-key")

        # Need to init components first
        model._init_components()
        model.add_expert(expert, production_weight=0.5)

        # Expert should be added to solver
        assert len(model._solver.experts) == 1

    def test_remove_expert(self):
        """Test removing expert from model."""
        model = Model()
        expert1 = Expert(model_name="gpt-4")
        expert2 = Expert(model_name="claude-3")

        model._init_components()
        model.add_expert(expert1, production_weight=0.5)
        model.add_expert(expert2, production_weight=0.3)

        model.remove_expert(expert1)

        assert expert1 not in model._solver.experts
        assert expert2 in model._solver.experts

    def test_clear_experts(self):
        """Test clearing all experts from model."""
        model = Model()
        expert = Expert(model_name="gpt-4")

        model._init_components()
        model.add_expert(expert, production_weight=0.5)

        model.clear_experts()

        assert len(model._solver.experts) == 0

    def test_set_benchmark(self):
        """Test setting benchmark."""
        model = Model()
        benchmark = Benchmark(name="test_benchmark")

        model.set_benchmark(benchmark)

        assert model.get_benchmark() == benchmark
        assert model.get_benchmark().name == "test_benchmark"

    def test_set_benchmark_create_new(self):
        """Test creating new benchmark via set_benchmark."""
        from autotrain.benchmark import Benchmark

        model = Model()
        benchmark = Benchmark(name="auto_benchmark")
        model.set_benchmark(benchmark)

        assert benchmark.name == "auto_benchmark"
        assert model.get_benchmark() == benchmark

    def test_add_benchmark_sample(self):
        """Test adding benchmark sample."""
        model = Model()
        model.add_benchmark_sample(
            input_data="What is 2+2?",
            expected_output="4",
            mode="exact_match",
        )

        benchmark = model.get_benchmark()
        assert benchmark is not None
        assert benchmark.sample_count == 1

    def test_checkpoint_methods(self, temp_dir):
        """Test checkpoint save/load methods."""
        model = Model(checkpoint_dir=str(temp_dir))

        # Should not raise
        model.save_checkpoint(iteration=0, metadata={"test": True})

        checkpoints = model.list_checkpoints()
        assert len(checkpoints) == 1

    def test_train_no_experts(self, mock_unsloth, mock_trainer, temp_dir):
        """Test training without experts."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        # Add initial samples
        samples = [
            Sample(input_data="input1", output_data="output1"),
            Sample(input_data="input2", output_data="output2"),
        ]

        # Mock the fine-tune method
        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
            initial_samples=samples,
        )

        assert summary["iterations_completed"] == 1
        assert summary["total_samples"] >= 2

    def test_train_with_experts(self, mock_unsloth, mock_trainer, mock_litellm, temp_dir):
        """Test training with experts."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        expert = Expert(model_name="gpt-4", api_key="test-key")

        # Mock the fine-tune method
        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
            experts=[(expert, 0.5)],
        )

        assert summary["iterations_completed"] == 1

    def test_train_with_benchmark(self, mock_unsloth, mock_trainer, temp_dir):
        """Test training with benchmark evaluation."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        benchmark = Benchmark(name="test")
        benchmark.add_sample(
            input_data="test input",
            expected_output="test output",
            evaluation_mode=EvaluationMode.EXACT_MATCH,
        )

        # Mock the fine-tune method
        model._fine_tune = MagicMock()

        summary = model.train(
            k=2,
            i=1,
            benchmark=benchmark,
        )

        assert "benchmark_history" in summary
        assert summary["iterations_completed"] == 1

    def test_train_early_stopping(self, mock_unsloth, mock_trainer, temp_dir):
        """Test training with early stopping."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        benchmark = Benchmark(name="test")
        benchmark.add_sample(
            input_data="test input",
            expected_output="test output",
        )

        # Mock the fine-tune method
        model._fine_tune = MagicMock()

        # Train for multiple iterations
        summary = model.train(
            k=2,
            i=5,
            benchmark=benchmark,
            early_stopping=True,
            early_stopping_patience=2,
        )

        # Should have benchmark history
        assert "benchmark_history" in summary

    def test_train_checkpoint_every(self, mock_unsloth, mock_trainer, temp_dir):
        """Test training with checkpoint_every."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        model._fine_tune = MagicMock()

        model.train(
            k=2,
            i=3,
            checkpoint_every=2,
        )

        # Should have saved checkpoints
        checkpoints = model.list_checkpoints()
        assert len(checkpoints) > 0

    def test_resume_from_checkpoint(self, mock_unsloth, mock_trainer, temp_dir):
        """Test resuming training from checkpoint."""
        model = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )

        model._fine_tune = MagicMock()

        # First training run
        model.train(k=2, i=1, checkpoint_every=1)

        # Create new model instance and resume
        model2 = Model(
            checkpoint_dir=str(temp_dir),
            sample_multiplier=1,
        )
        model2._fine_tune = MagicMock()

        summary = model2.train(
            k=2,
            i=1,
            resume_from_checkpoint=True,
        )

        assert summary["iterations_completed"] >= 1

    def test_generate_prompts(self):
        """Test that model has correct prompts after init."""
        model = Model()
        model._init_components()

        assert model._producer is not None
        assert model._solver is not None
        assert model._splitter is not None
        assert model._reviewer is not None

    def test_enable_checker(self):
        """Test model with checker enabled."""
        model = Model(enable_checker=True)
        model._init_components()

        assert model._checker is not None

    def test_disable_checker(self):
        """Test model with checker disabled."""
        model = Model(enable_checker=False)
        model._init_components()

        assert model._checker is None


class TestModelToolMethods:
    """Tests for Model tool management methods."""

    def test_add_tool_from_callable(self):
        """Test adding tool to Model from callable."""
        model = Model()

        def custom_func(x: int) -> int:
            return x * 2

        tool = model.add_tool(custom_func)
        assert model.has_tool("custom_func")
        assert tool.name == "custom_func"

    def test_add_tool_from_builtin(self):
        """Test adding built-in tool to Model."""
        from autotrain.tools import python, terminal

        model = Model()
        model.add_tool(python)
        model.add_tool(terminal)

        assert model.has_tool("python")
        assert model.has_tool("terminal")
        assert "python" in model.list_tools()
        assert "terminal" in model.list_tools()

    def test_remove_tool(self):
        """Test removing tool from Model."""
        from autotrain.tools import python

        model = Model()
        model.add_tool(python)

        assert model.has_tool("python")
        model.remove_tool("python")
        assert not model.has_tool("python")

    def test_clear_tools(self):
        """Test clearing all tools from Model."""
        from autotrain.tools import python, terminal

        model = Model()
        model.add_tool(python)
        model.add_tool(terminal)

        assert len(model.list_tools()) == 2
        model.clear_tools()
        assert len(model.list_tools()) == 0

    def test_get_tool(self):
        """Test getting tool from Model."""
        from autotrain.tools import terminal

        model = Model()
        model.add_tool(terminal)

        tool = model.get_tool("terminal")
        assert tool is not None
        assert tool.name == "terminal"

    def test_create_tool_decorator(self):
        """Test create_tool decorator on Model."""
        model = Model()

        @model.create_tool(description="Double a number")
        def double(x: int) -> int:
            return x * 2

        model.add_tool(double)
        assert model.has_tool("double")
        tool = model.get_tool("double")
        assert tool(5) == 10
