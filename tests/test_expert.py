"""Tests for Expert class."""

import json
from unittest.mock import MagicMock

import pytest
from autotrain import InferenceConfig, Sample

from autotrain.expert import Expert, ExpertPrompts


class TestExpertPrompts:
    """Tests for ExpertPrompts dataclass."""

    def test_default_prompts(self):
        """Test default expert prompt values."""
        prompts = ExpertPrompts()

        # Check that default values are None (new system uses getters)
        assert prompts.produce is None
        assert prompts.solve is None
        assert prompts.select is None
        assert prompts.review is None
        assert prompts.check is None

        # Check getter methods return detailed prompts
        produce_prompt = prompts.get_produce("test topic")
        assert "You are an expert data generator" in produce_prompt
        assert "test topic" in produce_prompt

        solve_prompt = prompts.get_solve()
        assert "You are an expert problem solver" in solve_prompt

        select_prompt = prompts.get_select()
        assert "You are an expert at selecting" in select_prompt

        review_prompt = prompts.get_review()
        assert "You are an expert reviewer" in review_prompt

        check_prompt = prompts.get_check()
        assert "You are an expert verifier" in check_prompt

    def test_custom_prompts(self):
        """Test custom expert prompts."""
        prompts = ExpertPrompts(
            produce="Custom produce prompt",
            solve="Custom solve prompt",
        )

        assert prompts.produce == "Custom produce prompt"
        assert prompts.solve == "Custom solve prompt"
        assert prompts.select is None
        assert prompts.review is None
        assert prompts.check is None

        # Custom prompts should be returned by getters
        assert prompts.get_produce("test topic") == "Custom produce prompt"
        assert prompts.get_solve() == "Custom solve prompt"
        # Default constants for others
        assert "You are an expert at selecting" in prompts.get_select()


class TestExpert:
    """Tests for Expert class."""

    def test_expert_init_default(self):
        """Test Expert initialization with defaults."""
        expert = Expert()

        assert expert.model_name == "unsloth/Qwen3.5-27B-GGUF"
        assert expert.production_rate == 0.5
        assert expert.api_key is None
        assert expert.api_base is None
        assert expert._cache_enabled is False
        assert expert._samples == []

    def test_expert_init_custom(self):
        """Test Expert initialization with custom params."""
        config = InferenceConfig(temperature=0.9)
        prompts = ExpertPrompts(produce="Custom")

        expert = Expert(
            model_name="unsloth/Qwen3.5-27B-GGUF",
            production_rate=0.75,
            inference_config=config,
            prompts=prompts,
            api_key="test-key",
            api_base="https://custom.api.com",
        )

        assert expert.model_name == "unsloth/Qwen3.5-27B-GGUF"
        assert expert.production_rate == 0.75
        assert expert.api_key == "test-key"
        assert expert.api_base == "https://custom.api.com"
        assert expert.inference_config.temperature == 0.9
        assert expert.prompts.produce == "Custom"

    def test_expert_produce(self, mock_litellm):
        """Test Expert.produce generates samples."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content='{"input": "What is 2+2?", "output": "4"}'))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        samples = expert.produce(task="math problems", count=3)

        assert len(samples) == 3
        assert all(isinstance(s, Sample) for s in samples)
        assert mock_litellm.call_count == 3

    def test_expert_produce_metadata(self, mock_litellm):
        """Test Expert.produce sample metadata."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content='{"input": "test", "output": "result"}'))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        samples = expert.produce(task="test", count=1)

        assert samples[0].metadata["source"] == "expert"

    def test_expert_solve(self, mock_litellm):
        """Test Expert.solve solves problems."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.solve("What is 2+2?")

        assert result == "Mocked response from LLM"
        mock_litellm.assert_called_once()

    def test_expert_solve_with_system_prompt(self, mock_litellm):
        """Test Expert.solve with custom system prompt."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        expert.solve(
            "What is 2+2?",
            system_prompt="You are a math expert",
        )

        assert mock_litellm.called
        call_args = mock_litellm.call_args
        messages = call_args[1]["messages"]

        # Check system prompt was included
        assert any(m["role"] == "system" for m in messages)

    def test_expert_solve_with_temperature(self, mock_litellm):
        """Test Expert.solve with custom temperature."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        expert.solve("What is 2+2?", temperature=0.9)

        call_args = mock_litellm.call_args
        assert call_args[1]["temperature"] == 0.9

    def test_expert_solve_batch(self, mock_litellm):
        """Test Expert.solve_batch solves multiple problems."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        results = expert.solve_batch(["Problem 1", "Problem 2", "Problem 3"])

        assert len(results) == 3
        assert mock_litellm.call_count == 3

    def test_expert_select(self, mock_litellm):
        """Test Expert.select chooses from options."""
        mock_litellm.return_value.choices = [MagicMock(message=MagicMock(content="2"))]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        samples = ["Option 1", "Option 2", "Option 3"]
        selected = expert.select(samples)

        assert selected == "Option 2"

    def test_expert_select_default(self, mock_litellm):
        """Test Expert.select returns first on parse failure."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content="invalid response"))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        samples = ["Option 1", "Option 2"]
        selected = expert.select(samples)

        # Should return first on parse failure
        assert selected == "Option 1"

    def test_expert_review(self, mock_litellm):
        """Test Expert.review provides feedback."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content="Score: 8/10. Good response."))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        sample = Sample(input_data="input", output_data="output")
        review = expert.review(sample)

        assert "score" in review
        assert "feedback" in review
        assert review["reviewer"] == "expert"

    def test_expert_review_with_criteria(self, mock_litellm):
        """Test Expert.review with custom criteria."""
        mock_litellm.return_value.choices = [MagicMock(message=MagicMock(content="Score: 9/10"))]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        sample = Sample(input_data="input", output_data="output")
        expert.review(sample, criteria="accuracy and clarity")

        assert mock_litellm.called

    def test_expert_check(self, mock_litellm):
        """Test Expert.check verifies correctness."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content="YES, this is correct"))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.check("What is 2+2?", "4")

        assert result["is_correct"] is True
        assert result["checker"] == "expert"

    def test_expert_check_incorrect(self, mock_litellm):
        """Test Expert.check detects incorrect solution."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content="NO, this is incorrect"))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.check("What is 2+2?", "5")

        assert result["is_correct"] is False

    def test_expert_check_strict(self, mock_litellm):
        """Test Expert.check with strict mode."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.check("input", "output", strict=True)

        assert result["strict"] is True

    def test_expert_compare(self, mock_litellm):
        """Test Expert.compare compares two outputs."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content="A is better because..."))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.compare("input", "output A", "output B")

        assert result["winner"] == "a"
        assert "explanation" in result

    def test_expert_compare_tie(self, mock_litellm):
        """Test Expert.compare detects tie."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content="TIE - both are equally good"))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.compare("input", "output A", "output B")

        assert result["winner"] == "tie"

    def test_expert_rate(self, mock_litellm):
        """Test Expert.rate provides numerical rating."""
        mock_litellm.return_value.choices = [MagicMock(message=MagicMock(content="Score: 8/10"))]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        result = expert.rate("input", "output", scale=10)

        assert "score" in result
        assert result["max_score"] == 10
        assert "normalized_score" in result

    def test_expert_rate_with_criteria(self, mock_litellm):
        """Test Expert.rate with multiple criteria."""
        mock_litellm.return_value.choices = [MagicMock(message=MagicMock(content="Score: 9/10"))]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        expert.rate(
            "input",
            "output",
            scale=10,
            criteria=["accuracy", "clarity", "completeness"],
        )

        assert mock_litellm.called

    def test_expert_cache_enable(self):
        """Test Expert.enable_cache."""
        expert = Expert()
        expert.enable_cache()

        assert expert._cache_enabled is True

    def test_expert_cache_disable(self):
        """Test Expert.disable_cache."""
        expert = Expert()
        expert.enable_cache()
        expert.disable_cache()

        assert expert._cache_enabled is False

    def test_expert_cache_clear(self):
        """Test Expert.clear_cache."""
        expert = Expert()
        expert.enable_cache()
        expert._cache["test"] = "value"
        expert.clear_cache()

        assert expert._cache == {}

    def test_expert_caching(self, mock_litellm):
        """Test Expert caches responses when enabled."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        expert.enable_cache()

        # First call
        expert.solve("Same question")
        # Second call with same question
        expert.solve("Same question")

        # Should only call API once
        assert mock_litellm.call_count == 1

    def test_expert_add_sample(self):
        """Test Expert.add_sample."""
        expert = Expert()
        expert.add_sample("input", "output", metadata={"test": True})

        assert len(expert._samples) == 1
        assert expert._samples[0].input_data == "input"
        assert expert._samples[0].metadata["test"] is True

    def test_expert_get_samples(self):
        """Test Expert.get_samples returns copy."""
        expert = Expert()
        expert.add_sample("input1", "output1")
        expert.add_sample("input2", "output2")

        samples = expert.get_samples()

        assert len(samples) == 2
        # Should be a copy
        samples.clear()
        assert len(expert.get_samples()) == 2

    def test_expert_clear_samples(self):
        """Test Expert.clear_samples."""
        expert = Expert()
        expert.add_sample("input1", "output1")
        expert.add_sample("input2", "output2")

        expert.clear_samples()

        assert len(expert.get_samples()) == 0

    def test_expert_set_production_rate(self):
        """Test Expert.set_production_rate."""
        expert = Expert()
        expert.set_production_rate(0.8)

        assert expert.production_rate == 0.8

    def test_expert_set_production_rate_invalid(self):
        """Test Expert.set_production_rate with invalid value."""
        expert = Expert()

        with pytest.raises(ValueError, match="must be between"):
            expert.set_production_rate(1.5)

        with pytest.raises(ValueError, match="must be between"):
            expert.set_production_rate(-0.1)

    def test_expert_set_inference_config(self):
        """Test Expert.set_inference_config."""
        expert = Expert()
        expert.set_inference_config(
            temperature=0.9,
            max_tokens=2048,
            top_p=0.95,
        )

        assert expert.inference_config.temperature == 0.9
        assert expert.inference_config.max_tokens == 2048
        assert expert.inference_config.top_p == 0.95

    def test_expert_get_info(self):
        """Test Expert.get_info."""
        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", production_rate=0.75)
        expert.add_sample("input", "output")

        info = expert.get_info()

        assert info["model_name"] == "unsloth/Qwen3.5-27B-GGUF"
        assert info["production_rate"] == 0.75
        assert info["samples_count"] == 1
        assert "cache_enabled" in info

    def test_expert_export_dataset_json(self, temp_dir, mock_litellm):
        """Test Expert.export_dataset in JSON format."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content='{"input": "test", "output": "result"}'))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        expert.produce(task="test", count=2)

        output_path = temp_dir / "dataset.json"
        expert.export_dataset(str(output_path), format="json")

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data) == 2

    def test_expert_export_dataset_jsonl(self, temp_dir, mock_litellm):
        """Test Expert.export_dataset in JSONL format."""
        mock_litellm.return_value.choices = [
            MagicMock(message=MagicMock(content='{"input": "test", "output": "result"}'))
        ]

        expert = Expert(model_name="unsloth/Qwen3.5-27B-GGUF", api_key="test-key")
        expert.produce(task="test", count=2)

        output_path = temp_dir / "dataset.jsonl"
        expert.export_dataset(str(output_path), format="jsonl")

        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_expert_import_dataset_json(self, temp_dir):
        """Test Expert.import_dataset in JSON format."""
        expert = Expert()

        # Create test dataset
        data = [
            {"input": "input1", "output": "output1"},
            {"input": "input2", "output": "output2"},
        ]

        input_path = temp_dir / "import.json"
        with open(input_path, "w") as f:
            json.dump(data, f)

        expert.import_dataset(str(input_path), format="json")

        assert len(expert.get_samples()) == 2

    def test_expert_import_dataset_jsonl(self, temp_dir):
        """Test Expert.import_dataset in JSONL format."""
        expert = Expert()

        # Create test dataset
        input_path = temp_dir / "import.jsonl"
        with open(input_path, "w") as f:
            f.write('{"input": "input1", "output": "output1"}\n')
            f.write('{"input": "input2", "output": "output2"}\n')

        expert.import_dataset(str(input_path), format="jsonl")

        assert len(expert.get_samples()) == 2

    def test_expert_import_dataset_not_found(self, temp_dir):
        """Test Expert.import_dataset with missing file."""
        expert = Expert()

        with pytest.raises(FileNotFoundError):
            expert.import_dataset(str(temp_dir / "nonexistent.json"))

    def test_expert_parse_sample_json(self):
        """Test Expert._parse_sample_response with JSON."""
        expert = Expert()
        response = '{"input": "What is 2+2?", "output": "4"}'

        sample = expert._parse_sample_response(response)

        assert sample.input_data == "What is 2+2?"
        assert sample.output_data == "4"

    def test_expert_parse_sample_input_output(self):
        """Test Expert._parse_sample_response with Input/Output format."""
        expert = Expert()
        response = """
        Input: What is the capital of France?
        Output: Paris is the capital of France.
        """

        sample = expert._parse_sample_response(response)

        assert "capital" in sample.input_data
        assert "Paris" in sample.output_data

    def test_expert_parse_sample_fallback(self):
        """Test Expert._parse_sample_response fallback."""
        expert = Expert()
        response = "This is just a plain response without structure"

        sample = expert._parse_sample_response(response)

        assert sample.input_data == ""
        assert sample.output_data == response


class TestExpertToolMethods:
    """Tests for Expert tool management methods."""

    def test_expert_add_tool(self):
        """Test adding tool to Expert."""
        from autotrain.tools import python, web_search

        expert = Expert()
        expert.add_tool(python)
        expert.add_tool(web_search)

        assert expert.has_tool("python")
        assert expert.has_tool("web_search")

    def test_expert_remove_tool(self):
        """Test removing tool from Expert."""
        from autotrain.tools import python

        expert = Expert()
        expert.add_tool(python)

        assert expert.has_tool("python")
        expert.remove_tool("python")
        assert not expert.has_tool("python")

    def test_expert_clear_tools(self):
        """Test clearing all tools from Expert."""
        from autotrain.tools import python, web_search

        expert = Expert()
        expert.add_tool(python)
        expert.add_tool(web_search)

        assert len(expert.list_tools()) == 2
        expert.clear_tools()
        assert len(expert.list_tools()) == 0

    def test_expert_list_tools(self):
        """Test listing tools on Expert."""
        from autotrain.tools import python, web_search

        expert = Expert()
        expert.add_tool(python)
        expert.add_tool(web_search)

        tools = expert.list_tools()
        assert "python" in tools
        assert "web_search" in tools

    def test_expert_get_tool(self):
        """Test getting tool from Expert."""
        from autotrain.tools import web_search

        expert = Expert()
        expert.add_tool(web_search)

        tool = expert.get_tool("web_search")
        assert tool is not None
        assert tool.name == "web_search"

    def test_expert_get_tool_schemas(self):
        """Test getting tool schemas from Expert."""
        from autotrain.tools import python, web_search

        expert = Expert()
        expert.add_tool(python)
        expert.add_tool(web_search)

        schemas = expert.get_tool_schemas()
        assert len(schemas) == 2

    def test_expert_tool_info(self):
        """Test Expert info includes tools."""
        from autotrain.tools import python

        expert = Expert(model_name="test-model")
        expert.add_tool(python)

        info = expert.get_info()
        assert "tools" in info
        assert "python" in info["tools"]
