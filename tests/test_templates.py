"""Tests for instruction tuning templates."""

import pytest

from autotrain.templates import (
    InstructionTemplate,
    TemplateType,
    TEMPLATE_REGISTRY,
    apply_chat_template,
    auto_detect_template,
    create_custom_template,
    format_sample,
    format_samples_batch,
    get_alpaca_template,
    get_chatml_template,
    get_llama3_template,
    get_llama31_template,
    get_llama32_template,
    get_llama33_template,
    get_gemma_template,
    get_gemma2_template,
    get_gemma3_template,
    get_phi_template,
    get_phi3_template,
    get_phi4_template,
    get_qwen25_template,
    get_zephyr_template,
    get_vicuna_template,
    get_template,
    standardize_sharegpt,
    get_chat_template,
)


class TestInstructionTemplate:
    def test_template_creation(self):
        template = InstructionTemplate(
            name="test", user_template="{instruction}", assistant_template="{output}"
        )
        assert template.name == "test"

    def test_format_basic(self):
        template = InstructionTemplate(
            name="test", user_template="Q:{instruction}A:", assistant_template="{output}"
        )
        result = template.format(instruction="2+2?", output="4")
        assert "2+2?" in result and "4" in result

    def test_format_prompt_only(self):
        template = InstructionTemplate(
            name="test", user_template="Q:{instruction}", assistant_template="{output}"
        )
        result = template.format_prompt(instruction="test")
        assert "output" not in result.lower() or "{output}" not in result


class TestPredefinedTemplates:
    def test_alpaca_template(self):
        template = get_alpaca_template()
        assert template.name == "alpaca"
        assert "Instruction" in template.user_template

    def test_chatml_template(self):
        template = get_chatml_template()
        assert template.name == "chatml"

    def test_llama3_template(self):
        template = get_llama3_template()
        assert template.name == "llama3"


class TestGetTemplate:
    def test_get_template_valid(self):
        template = get_template("alpaca")
        assert template.name == "alpaca"

    def test_get_template_invalid(self):
        with pytest.raises(ValueError):
            get_template("nonexistent")


class TestAutoDetectTemplate:
    def test_detect_llama3(self):
        template = auto_detect_template("meta-llama/Llama-3-8b")
        assert template.name == "llama3"

    def test_detect_mistral(self):
        template = auto_detect_template("mistralai/Mistral-7B")
        assert template.name == "mistral"

    def test_detect_default(self):
        template = auto_detect_template("unknown-model")
        assert template.name == "alpaca"


class TestCreateCustomTemplate:
    def test_create_custom(self):
        template = create_custom_template(
            name="my_template",
            user_template="User: {instruction}",
            assistant_template="Assistant: {output}",
        )
        assert template.name == "my_template"
        assert template.template_type == TemplateType.CUSTOM


class TestFormatSample:
    def test_format_sample_basic(self):
        result = format_sample(instruction="test", output="result")
        assert "test" in result and "result" in result

    def test_format_sample_with_model(self):
        result = format_sample(instruction="test", output="result", model_name="llama-3")
        assert "test" in result


class TestFormatSamplesBatch:
    def test_format_batch(self):
        samples = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
        ]
        results = format_samples_batch(samples)
        assert len(results) == 2
        assert any("Q1" in r for r in results)
        assert any("A1" in r for r in results)


class TestApplyChatTemplate:
    def test_apply_chat_basic(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = apply_chat_template(messages)
        assert "Hello" in result

    def test_apply_chat_with_system(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_chat_template(messages)
        assert "You are helpful" in result or "system" in result.lower()

    def test_apply_chat_with_role_mapping(self):
        messages = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ]
        result = apply_chat_template(messages, role_mapping={"human": "user", "gpt": "assistant"})
        assert "Hello" in result

    def test_apply_chat_add_generation_prompt(self):
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = apply_chat_template(messages, add_generation_prompt=True)
        assert "Hello" in result

    def test_apply_chat_tokenize(self):
        class MockTokenizer:
            def encode(self, text, **kwargs):
                return [1, 2, 3]

        messages = [{"role": "user", "content": "Hello"}]
        tokenizer = MockTokenizer()
        result = apply_chat_template(messages, tokenize=True, tokenizer=tokenizer)
        assert result == [1, 2, 3]

    def test_apply_chat_tokenize_no_tokenizer(self):
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ValueError, match="tokenizer must be provided"):
            apply_chat_template(messages, tokenize=True)


class TestNewTemplates:
    def test_llama31_template(self):
        template = get_llama31_template()
        assert template.name == "llama-3.1"

    def test_llama32_template(self):
        template = get_llama32_template()
        assert template.name == "llama-3.2"

    def test_llama33_template(self):
        template = get_llama33_template()
        assert template.name == "llama-3.3"

    def test_gemma2_template(self):
        template = get_gemma2_template()
        assert template.name == "gemma2"

    def test_gemma3_template(self):
        template = get_gemma3_template()
        assert template.name == "gemma3"

    def test_phi3_template(self):
        template = get_phi3_template()
        assert template.name == "phi-3"

    def test_phi4_template(self):
        template = get_phi4_template()
        assert template.name == "phi-4"

    def test_qwen25_template(self):
        template = get_qwen25_template()
        assert template.name == "qwen-2.5"

    def test_zephyr_template(self):
        template = get_zephyr_template()
        assert template.name == "zephyr"

    def test_vicuna_template(self):
        template = get_vicuna_template()
        assert template.name == "vicuna"

    def test_template_registry_has_all(self):
        assert "alpaca" in TEMPLATE_REGISTRY
        assert "llama3" in TEMPLATE_REGISTRY
        assert "llama-3.1" in TEMPLATE_REGISTRY
        assert "gemma3" in TEMPLATE_REGISTRY
        assert "phi-4" in TEMPLATE_REGISTRY
        assert "qwen-2.5" in TEMPLATE_REGISTRY


class TestAutoDetectNewModels:
    def test_detect_llama31(self):
        template = auto_detect_template("meta-llama/Llama-3.1-8B")
        assert template.name == "llama-3.1"

    def test_detect_llama32(self):
        template = auto_detect_template("meta-llama/Llama-3.2-1B")
        assert template.name == "llama-3.2"

    def test_detect_llama33(self):
        template = auto_detect_template("meta-llama/Llama-3.3-70B")
        assert template.name == "llama-3.3"

    def test_detect_gemma3(self):
        template = auto_detect_template("google/gemma-3-4b")
        assert template.name == "gemma3"

    def test_detect_phi4(self):
        template = auto_detect_template("microsoft/phi-4")
        assert template.name == "phi-4"

    def test_detect_qwen25(self):
        template = auto_detect_template("Qwen/Qwen2.5-7B")
        assert template.name == "qwen-2.5"


class TestStandardizeShareGPT:
    def test_standardize_basic(self):
        data = [
            [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
            ]
        ]
        result = standardize_sharegpt(data)
        assert result[0][0]["role"] == "user"
        assert result[0][0]["content"] == "Hi"
        assert result[0][1]["role"] == "assistant"
        assert result[0][1]["content"] == "Hello"

    def test_standardize_custom_mapping(self):
        data = [
            [
                {"from": "user", "value": "Hi"},
                {"from": "assistant", "value": "Hello"},
            ]
        ]
        result = standardize_sharegpt(data, role_mapping={"user": "user", "assistant": "assistant"})
        assert result[0][0]["role"] == "user"


class TestGetChatTemplate:
    def test_get_chat_template_basic(self):
        class MockTokenizer:
            def __init__(self):
                self.chat_template = None

        tokenizer = MockTokenizer()
        result = get_chat_template(tokenizer, chat_template="alpaca")
        assert result.chat_template is not None
