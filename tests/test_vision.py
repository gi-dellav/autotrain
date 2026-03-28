"""Tests for Vision components."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


class TestVisionSample:
    """Tests for VisionSample dataclass."""

    def test_vision_sample_init_default(self):
        """Test VisionSample initialization with defaults."""
        from autotrain.data_types import VisionSample

        sample = VisionSample(
            input_data="What is in this image?",
            output_data="A cat sitting on a couch",
        )

        assert sample.input_data == "What is in this image?"
        assert sample.output_data == "A cat sitting on a couch"
        assert sample.images == []
        assert sample.metadata == {}

    def test_vision_sample_init_with_images(self):
        """Test VisionSample initialization with images."""
        from autotrain.data_types import VisionSample

        sample = VisionSample(
            input_data="Describe this image",
            output_data="A sunset over the ocean",
            images=["image1.jpg", "image2.jpg"],
            metadata={"source": "test"},
        )

        assert sample.images == ["image1.jpg", "image2.jpg"]
        assert sample.metadata == {"source": "test"}

    def test_vision_sample_to_dict(self):
        """Test VisionSample to_dict method."""
        from autotrain.data_types import VisionSample

        sample = VisionSample(
            input_data="Test input",
            output_data="Test output",
            images=["img1.png"],
            metadata={"key": "value"},
        )

        result = sample.to_dict()

        assert result["input_data"] == "Test input"
        assert result["output_data"] == "Test output"
        assert result["images"] == ["img1.png"]
        assert result["metadata"] == {"key": "value"}

    def test_vision_sample_from_dict(self):
        """Test VisionSample from_dict classmethod."""
        from autotrain.data_types import VisionSample

        data = {
            "input_data": "Test input",
            "output_data": "Test output",
            "images": ["img1.png", "img2.png"],
            "metadata": {"key": "value"},
        }

        sample = VisionSample.from_dict(data)

        assert sample.input_data == "Test input"
        assert sample.output_data == "Test output"
        assert sample.images == ["img1.png", "img2.png"]
        assert sample.metadata == {"key": "value"}

    def test_vision_sample_to_conversation(self):
        """Test VisionSample to_conversation method."""
        from autotrain.data_types import VisionSample

        sample = VisionSample(
            input_data="What do you see?",
            output_data="I see a dog",
            images=["dog.jpg"],
        )

        conversation = sample.to_conversation()

        assert len(conversation) == 2
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        assert conversation[1]["content"][0]["text"] == "I see a dog"

    def test_vision_sample_inherits_from_sample(self):
        """Test that VisionSample inherits from Sample."""
        from autotrain.data_types import Sample, VisionSample

        sample = VisionSample(
            input_data="test",
            output_data="test",
        )

        assert isinstance(sample, Sample)


class TestVisionExpert:
    """Tests for VisionExpert class."""

    def test_vision_expert_init_default(self):
        """Test VisionExpert initialization with defaults."""
        from autotrain.expert import VisionExpert

        expert = VisionExpert()

        assert expert.model_name == "qwen/qwen3.5-397b-a17b"
        assert expert.production_rate == 0.5
        assert expert.vision_model is True
        assert expert._vision_samples == []

    def test_vision_expert_init_custom(self):
        """Test VisionExpert initialization with custom params."""
        from autotrain.expert import VisionExpert
        from autotrain.config import InferenceConfig
        from autotrain.expert import ExpertPrompts

        config = InferenceConfig(temperature=0.9)
        prompts = ExpertPrompts(produce="Custom vision produce")

        expert = VisionExpert(
            model_name="qwen/qwen2.5-vl-7b-instruct",
            production_rate=0.75,
            inference_config=config,
            prompts=prompts,
            api_key="test-key",
        )

        assert expert.model_name == "qwen/qwen2.5-vl-7b-instruct"
        assert expert.production_rate == 0.75
        assert expert.vision_model is True
        assert expert.api_key == "test-key"
        assert expert.inference_config.temperature == 0.9
        assert expert.prompts.produce == "Custom vision produce"

    def test_vision_expert_get_info(self):
        """Test VisionExpert.get_info method."""
        from autotrain.expert import VisionExpert

        expert = VisionExpert(model_name="test-model", production_rate=0.5)
        expert._vision_samples = [MagicMock(), MagicMock()]

        info = expert.get_info()

        assert info["vision_model"] is True
        assert info["vision_samples_count"] == 2
        assert info["model_name"] == "test-model"

    def test_vision_expert_add_sample(self):
        """Test adding vision samples to VisionExpert."""
        from autotrain.expert import VisionExpert
        from autotrain.data_types import VisionSample

        expert = VisionExpert()
        sample = VisionSample(
            input_data="test",
            output_data="test",
            images=["img.jpg"],
        )

        expert.add_sample(sample)

        assert len(expert._samples) == 1
        assert len(expert._vision_samples) == 1

    def test_vision_expert_clear_vision_samples(self):
        """Test clearing vision samples."""
        from autotrain.expert import VisionExpert

        expert = VisionExpert()
        expert._vision_samples = [MagicMock(), MagicMock()]

        expert.clear_vision_samples()

        assert expert._vision_samples == []


class TestVisionProducer:
    """Tests for VisionProducer class."""

    def test_vision_producer_init(self):
        """Test VisionProducer initialization."""
        from autotrain.components.vision_producer import VisionProducer
        from autotrain.config import InferenceConfig

        mock_model = MagicMock()
        mock_model.is_loaded = False

        producer = VisionProducer(
            model=mock_model,
            prompt="Test prompt",
            inference_config=InferenceConfig(),
        )

        assert producer.model == mock_model
        assert producer.prompt == "Test prompt"

    def test_vision_producer_default_tasks(self):
        """Test default vision tasks are defined."""
        from autotrain.components.vision_producer import VisionProducer

        assert len(VisionProducer.DEFAULT_VISION_TASKS) > 0
        assert "Describe" in VisionProducer.DEFAULT_VISION_TASKS[0]

    def test_vision_producer_generate_with_images(self):
        """Test generating vision samples with images."""
        from autotrain.components.vision_producer import VisionProducer
        from autotrain.config import InferenceConfig
        from autotrain.data_types import VisionSample

        mock_model = MagicMock()
        mock_model.is_loaded = False
        mock_model.generate = MagicMock(return_value="Generated description")

        producer = VisionProducer(
            model=mock_model,
            prompt="Describe images",
            inference_config=InferenceConfig(),
        )

        images = ["img1.jpg", "img2.jpg"]
        samples = producer.generate(count=2, images=images)

        assert len(samples) == 2
        assert all(isinstance(s, VisionSample) for s in samples)
        assert samples[0].images == ["img1.jpg"]
        assert samples[1].images == ["img2.jpg"]

    def test_vision_producer_generate_without_images(self):
        """Test generating vision samples without images."""
        from autotrain.components.vision_producer import VisionProducer
        from autotrain.config import InferenceConfig
        from autotrain.data_types import VisionSample

        mock_model = MagicMock()
        mock_model.is_loaded = False

        producer = VisionProducer(
            model=mock_model,
            prompt="Test task",
            inference_config=InferenceConfig(),
        )

        samples = producer.generate(count=3)

        assert len(samples) == 3
        assert all(isinstance(s, VisionSample) for s in samples)
        assert all(s.images == [] for s in samples)

    def test_vision_producer_generate_with_prompts(self):
        """Test generating with custom prompts."""
        from autotrain.components.vision_producer import VisionProducer
        from autotrain.config import InferenceConfig

        mock_model = MagicMock()
        mock_model.is_loaded = False

        producer = VisionProducer(
            model=mock_model,
            prompt="Custom",
            inference_config=InferenceConfig(),
        )

        custom_prompts = ["Describe this image", "What colors are present?"]
        samples = producer.generate_with_prompts(count=2, custom_prompts=custom_prompts)

        assert len(samples) == 2
        assert samples[0].input_data == "Describe this image"
        assert samples[1].input_data == "What colors are present?"


class TestVisionDataCollator:
    """Tests for VisionDataCollator class."""

    def test_vision_collator_init_default(self):
        """Test VisionDataCollator initialization with defaults."""
        from autotrain.components.vision_collator import VisionDataCollator

        mock_processor = MagicMock()

        collator = VisionDataCollator(processor=mock_processor)

        assert collator.processor == mock_processor
        assert collator.max_seq_length == 2048
        assert collator.train_on_responses_only is False
        assert collator.completion_only_loss is True

    def test_vision_collator_init_custom(self):
        """Test VisionDataCollator initialization with custom params."""
        from autotrain.components.vision_collator import VisionDataCollator

        mock_processor = MagicMock()

        collator = VisionDataCollator(
            processor=mock_processor,
            max_seq_length=4096,
            train_on_responses_only=True,
            resize="max",
        )

        assert collator.max_seq_length == 4096
        assert collator.train_on_responses_only is True
        assert collator.resize == "max"

    def test_vision_collator_process_basic(self):
        """Test basic sample processing."""
        from autotrain.components.vision_collator import VisionDataCollator

        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        collator = VisionDataCollator(processor=mock_processor)

        sample = {
            "messages": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "A cat"},
            ],
            "images": ["img.jpg"],
        }

        result = collator.process_sample(sample)

        assert "input_ids" in result

    def test_vision_collator_format_dataset(self):
        """Test dataset formatting."""
        from autotrain.components.vision_collator import VisionDataCollator

        mock_processor = MagicMock()
        mock_processor.apply_chat_template = MagicMock(return_value="formatted text")
        mock_processor.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        collator = VisionDataCollator(processor=mock_processor)

        dataset = [
            {"messages": [{"role": "user", "content": "test1"}], "images": []},
            {"messages": [{"role": "user", "content": "test2"}], "images": []},
        ]

        result = collator.format_dataset(dataset)

        assert isinstance(result, list)


class TestVisionModel:
    """Tests for VisionModel class."""

    def test_vision_model_init_default(self):
        """Test VisionModel initialization with defaults."""
        from autotrain.core.vision_model import VisionModel

        model = VisionModel()

        assert model.model_name == "unsloth/Qwen3.5-35B-A3B-GGUF"
        assert model.sample_multiplier == 2
        assert model._is_model_loaded is False
        assert model._samples == []

    def test_vision_model_init_custom(self):
        """Test VisionModel initialization with custom params."""
        from autotrain.config import InferenceConfig
        from autotrain.core.vision_model import VisionModel
        from autotrain.config import Prompts

        config = InferenceConfig(temperature=0.9)
        prompts = Prompts(producer="Custom")

        model = VisionModel(
            model_name="unsloth/Qwen2.5-VL-7B-GGUF",
            sample_multiplier=3,
            inference_config=config,
            prompts=prompts,
        )

        assert model.model_name == "unsloth/Qwen2.5-VL-7B-GGUF"
        assert model.sample_multiplier == 3
        assert model.inference_config.temperature == 0.9
        assert model.prompts.producer == "Custom"

    def test_vision_model_add_sample(self):
        """Test adding samples to VisionModel."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.data_types import VisionSample

        model = VisionModel()
        sample = VisionSample(
            input_data="Test",
            output_data="Test output",
            images=["img.jpg"],
        )

        model.add_sample(sample)

        assert len(model._samples) == 1

    def test_vision_model_add_sample_from_sample(self):
        """Test adding Sample (non-vision) to VisionModel."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.data_types import Sample

        model = VisionModel()
        sample = Sample(input_data="Test", output_data="Test output")

        model.add_sample(sample)

        assert len(model._samples) == 1
        assert model._samples[0].input_data == "Test"

    def test_vision_model_get_samples(self):
        """Test getting samples from VisionModel."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.data_types import VisionSample

        model = VisionModel()
        sample1 = VisionSample(input_data="Test1", output_data="Out1")
        sample2 = VisionSample(input_data="Test2", output_data="Out2")

        model.add_sample(sample1)
        model.add_sample(sample2)

        samples = model.get_samples()

        assert len(samples) == 2

    def test_vision_model_clear_samples(self):
        """Test clearing samples."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.data_types import VisionSample

        model = VisionModel()
        model._samples = [MagicMock(), MagicMock()]

        model.clear_samples()

        assert model._samples == []

    def test_vision_model_add_expert(self):
        """Test adding expert to VisionModel."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.expert import VisionExpert

        model = VisionModel()
        expert = VisionExpert(model_name="test-expert")

        model.add_expert(expert, weight=0.5)

        experts = model.get_experts()
        assert len(experts) == 1
        assert experts[0][0].model_name == "test-expert"
        assert experts[0][1] == 0.5

    def test_vision_model_remove_expert(self):
        """Test removing expert from VisionModel."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.expert import VisionExpert

        model = VisionModel()
        expert = VisionExpert(model_name="test-expert")

        model.add_expert(expert, weight=0.5)
        model.remove_expert(expert)

        assert model.get_experts() == []

    def test_vision_model_clear_experts(self):
        """Test clearing all experts."""
        from autotrain.core.vision_model import VisionModel
        from autotrain.expert import VisionExpert

        model = VisionModel()
        expert1 = VisionExpert(model_name="expert1")
        expert2 = VisionExpert(model_name="expert2")

        model.add_expert(expert1, weight=0.3)
        model.add_expert(expert2, weight=0.3)
        model.clear_experts()

        assert model.get_experts() == []

    def test_vision_model_is_loaded_property(self):
        """Test is_loaded property."""
        from autotrain.core.vision_model import VisionModel

        model = VisionModel()

        assert model.is_loaded is False
        assert model.is_loaded == model._is_model_loaded

    def test_vision_model_processor_property(self):
        """Test processor property."""
        from autotrain.core.vision_model import VisionModel

        model = VisionModel()
        model._processor = MagicMock()

        assert model.processor == model._processor

    def test_vision_model_for_inference(self):
        """Test for_inference method."""
        import sys
        from types import ModuleType
        from autotrain.core.vision_model import VisionModel

        mock_unsloth = ModuleType("unsloth")
        mock_fv = MagicMock()
        mock_unsloth.FastVisionModel = mock_fv
        sys.modules["unsloth"] = mock_unsloth

        model = VisionModel()
        model._fast_model = MagicMock()

        model.for_inference()

        mock_fv.for_inference.assert_called_once_with(model._fast_model)

    def test_vision_model_repr(self):
        """Test VisionModel __repr__."""
        from autotrain.core.vision_model import VisionModel

        model = VisionModel(model_name="test/model")

        assert "VisionModel" in repr(model)
        assert "test/model" in repr(model)


class TestTraining:
    """Tests for vision training functions."""

    def test_train_vision_model_import(self):
        """Test that train_vision_model can be imported."""
        from autotrain.core.training import train_vision_model, _fine_tune_vision

        assert callable(train_vision_model)
        assert callable(_fine_tune_vision)

    def test_train_vision_model_in_exports(self):
        """Test that train_vision_model is exported."""
        from autotrain import train_vision_model

        assert callable(train_vision_model)
