"""Model expert management, inference, and export methods."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from autotrain.config import InferenceConfig

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.core.base_model import BaseModel
    from autotrain.expert import Expert


# ==================== Expert Management ====================


def add_expert(
    model: "BaseModel",
    expert: "Expert",
    production_weight: float = 1.0,
    check_weight: float = 0.0,
) -> None:
    """Add an expert with weighted production rates."""
    if model._solver:
        model._solver.add_expert(expert, weight=production_weight)

    if model._splitter:
        model._splitter.add_expert(expert)

    if model._checker and check_weight > 0:
        model._checker.add_expert(expert)


def remove_expert(model: "BaseModel", expert: "Expert") -> None:
    """Remove an expert from all components."""
    if model._solver:
        model._solver.remove_expert(expert)
    if model._splitter:
        model._splitter.remove_expert(expert)
    if model._checker:
        model._checker.remove_expert(expert)


def clear_experts(model: "BaseModel") -> None:
    """Clear all experts from all components."""
    if model._solver:
        model._solver.clear_experts()
    if model._splitter:
        model._splitter.experts.clear()
    if model._checker:
        model._checker.experts.clear()


def set_expert_weights(model: "BaseModel", weights: dict["Expert", float]) -> None:
    """Set production weights for multiple experts."""
    if model._solver:
        model._solver.clear_experts()
        for expert, weight in weights.items():
            model._solver.add_expert(expert, weight=weight)


def add_experts(
    model: "BaseModel",
    experts: list[tuple["Expert", float]],
    check_weight: float = 0.0,
) -> None:
    """Add multiple experts with production weights."""
    for expert, production_weight in experts:
        add_expert(
            model=model,
            expert=expert,
            production_weight=production_weight,
            check_weight=check_weight,
        )
    print(f"Added {len(experts)} experts with production weights")


# ==================== Benchmark Management ====================


def set_benchmark(
    model: "BaseModel",
    benchmark: Optional["Benchmark"] = None,
    name: str = "default",
    expert_model_name: Optional[str] = None,
    expert_api_key: Optional[str] = None,
) -> "Benchmark":
    """Set or create a benchmark for evaluation."""
    from autotrain.benchmark import Benchmark

    if benchmark:
        model._benchmark = benchmark
    else:
        model._benchmark = Benchmark(
            name=name, expert_model_name=expert_model_name, expert_api_key=expert_api_key
        )
    return model._benchmark


def get_benchmark(model: "BaseModel") -> Optional["Benchmark"]:
    """Get the current benchmark."""
    return model._benchmark


def add_benchmark_sample(
    model: "BaseModel",
    input_data: str,
    expected_output: str,
    mode: str = "exact_match",
) -> None:
    """Add a sample to the benchmark."""
    from autotrain.benchmark import Benchmark, EvaluationMode

    if not model._benchmark:
        model._benchmark = Benchmark()

    eval_mode = EvaluationMode.EXACT_MATCH if mode == "exact_match" else EvaluationMode.LLM_JUDGE
    model._benchmark.add_sample(input_data, expected_output, evaluation_mode=eval_mode)


def load_benchmark(model: "BaseModel", path: str, format: str = "json") -> "Benchmark":
    """Load a benchmark from file."""
    model._benchmark = Benchmark.load(path, format=format)
    return model._benchmark


# ==================== Checkpoint Management ====================


def save_checkpoint(
    model: "BaseModel",
    iteration: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Save a checkpoint."""
    if iteration is None:
        iteration = model._current_iteration

    benchmark_accuracy = None
    if model._benchmark and model._benchmark.best_metrics:
        benchmark_accuracy = model._benchmark.best_metrics.accuracy

    model._checkpoint_manager.save(
        model=model,  # type: ignore
        iteration=iteration,
        benchmark_accuracy=benchmark_accuracy,
        metadata=metadata,
    )


def load_checkpoint(
    model: "BaseModel",
    checkpoint_id: Optional[str] = None,
    iteration: Optional[int] = None,
) -> None:
    """Load a checkpoint."""
    model._checkpoint_manager.load(model=model, checkpoint_id=checkpoint_id, iteration=iteration)  # type: ignore


def restore_best_checkpoint(model: "BaseModel") -> None:
    """Restore the best checkpoint (by benchmark accuracy)."""
    model._checkpoint_manager.restore_best(model)  # type: ignore


def list_checkpoints(model: "BaseModel") -> list:
    """List all available checkpoints."""
    return model._checkpoint_manager.list_checkpoints()


# ==================== Inference ====================


def _call_model(
    model: "BaseModel", prompt: str, inference_config: Optional[InferenceConfig] = None
) -> str:
    """Generate output from the model."""
    if not model._is_model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Assert that model components are loaded for type checker
    assert model._tokenizer is not None, "Tokenizer should not be None when model is loaded"
    assert model._fast_model is not None, "Fast model should not be None when model is loaded"

    config = inference_config or model.inference_config

    try:
        inputs = model._tokenizer(prompt, return_tensors="pt").to(model._fast_model.device)

        outputs = model._fast_model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            thinking=config.thinking,
        )

        return model._tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Inference error: {e}")
        return f"[Error generating output: {e}]"


def generate(
    model: "BaseModel",
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> str:
    """Generate text from the model."""
    config = InferenceConfig(
        temperature=temperature if temperature is not None else model.inference_config.temperature,
        max_tokens=max_tokens if max_tokens is not None else model.inference_config.max_tokens,
        top_p=top_p if top_p is not None else model.inference_config.top_p,
    )
    return _call_model(model, prompt, config)


def _call_model_with_tools(
    model: "BaseModel",
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the model with tool calling support.

    Args:
        model: The Model instance
        messages: List of message dicts with role and content
        tools: Optional list of tool schemas
        tool_choice: "auto", "none", or specific tool name

    Returns:
        Dict with 'content' and optional 'tool_calls'
    """
    if not model._is_model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    assert model._tokenizer is not None, "Tokenizer should not be None when model is loaded"
    assert model._fast_model is not None, "Fast model should not be None when model is loaded"

    config = model.inference_config
    tool_schemas = tools or model._tools.get_schemas()

    try:
        from autotrain.templates import apply_chat_template, get_chat_template

        # Apply template to tokenizer (returns tokenizer)
        tokenizer = get_chat_template(
            model._tokenizer,
            chat_template="chatml",
        )

        # Apply chat template to messages
        formatted = apply_chat_template(
            messages,
            tokenizer=tokenizer,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = model._tokenizer(formatted, return_tensors="pt").to(model._fast_model.device)

        generation_kwargs = {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature,
            "do_sample": config.temperature > 0,
            "top_p": config.top_p,
        }

        if tool_schemas:
            generation_kwargs["tools"] = tool_schemas
            if tool_choice:
                generation_kwargs["tool_choice"] = tool_choice

        outputs = model._fast_model.generate(**inputs, **generation_kwargs)

        response_text = model._tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_content = (
            response_text.split("[/INST]")[-1].strip()
            if "[/INST]" in response_text
            else response_text
        )

        return {"content": response_content}

    except Exception as e:
        print(f"Inference error with tools: {e}")
        return {"content": f"[Error: {e}]", "tool_calls": []}


def _parse_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from model response.

    This is a simplified parser - in production you might want to use
    the model's built-in tool call parsing if available.
    """
    tool_calls = []

    json_patterns = [
        r'\[{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})',
        r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
    ]

    for pattern in json_patterns:
        matches = re.finditer(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                tool_name = match.group(1) if match.lastindex is not None and match.lastindex >= 1 else None
                args_str = match.group(2) if match.lastindex is not None and match.lastindex >= 2 else "{}"

                if tool_name:
                    tool_args = json.loads(args_str)
                    tool_calls.append(
                        {
                            "name": tool_name,
                            "arguments": tool_args,
                        }
                    )
            except (json.JSONDecodeError, AttributeError):
                continue

    python_pattern = r"```python\n(.*?)```"
    py_matches = re.finditer(python_pattern, response_text, re.DOTALL)
    for i, match in enumerate(py_matches):
        tool_calls.append(
            {
                "name": "python",
                "arguments": {"code": match.group(1).strip()},
                "type": "code",
            }
        )

    return tool_calls


def generate_with_tools(
    model: "BaseModel",
    prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    max_tool_calls: int = 10,
    tool_choice: Optional[str] = None,
) -> str:
    """
    Generate text with tool calling support.

    This function handles the full tool calling loop:
    1. Send prompt with tools to model
    2. Model may return tool calls
    3. Execute tools and feed results back
    4. Continue until no more tool calls

    Args:
        model: The Model instance
        prompt: User prompt
        tools: Optional list of tool schemas (uses registered tools if None)
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        top_p: Top-p sampling
        max_tool_calls: Maximum tool call iterations (default: 10)
        tool_choice: "auto", "none", or specific tool name

    Returns:
        Final text response after tool execution

    Example:
        from autotrain.tools import python, calculator

        model.add_tool(python)
        model.add_tool(calculator)

        result = model.generate_with_tools(
            "Calculate 15 * 23 and write fib(20)"
        )
    """
    import re

    if temperature is not None:
        model.inference_config.temperature = temperature
    if max_tokens is not None:
        model.inference_config.max_tokens = max_tokens
    if top_p is not None:
        model.inference_config.top_p = top_p

    tool_schemas = tools or model._tools.get_schemas()

    if not tool_schemas:
        return generate(model, prompt)

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    tool_call_count = 0
    final_content = ""

    while tool_call_count < max_tool_calls:
        response = _call_model_with_tools(
            model,
            messages,
            tools=tool_schemas,
            tool_choice=tool_choice,
        )

        content = response.get("content", "")
        final_content = content
        messages.append({"role": "assistant", "content": content})

        parsed_tool_calls = _parse_tool_calls(content)

        if not parsed_tool_calls:
            break

        for tool_call in parsed_tool_calls:
            tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("arguments") or tool_call.get("function", {}).get(
                "arguments", {}
            )

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {"code": tool_args}

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{tool_call_count}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                    ],
                }
            )

            try:
                tool_result = model._tools.execute_tool(tool_name, tool_args)
                result_str = str(tool_result)
            except Exception as e:
                result_str = f"Error executing tool '{tool_name}': {str(e)}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{tool_call_count}",
                    "content": result_str,
                }
            )

            tool_call_count += 1

        if tool_call_count >= max_tool_calls:
            final_content += f"\n\n[Max tool calls ({max_tool_calls}) reached]"
            break

    return final_content


# ==================== Export Methods ====================


# ==================== Export Methods ====================


def export_gguf(
    model: "BaseModel",
    output_path: str,
    quantization: str = "q4_k_m",
    merge_adapter: bool = True,
) -> str:
    """Export model to GGUF format."""
    if not model._is_model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    try:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert model._fast_model is not None, "Fast model should be loaded"
        if merge_adapter:
            merged_path = output_dir / "merged_model"
            model._fast_model.save_pretrained(str(merged_path), save_method="merged_16bit")
            model_path = str(merged_path)
        else:
            adapter_path = output_dir / "adapter"
            model._fast_model.save_pretrained(str(adapter_path))
            model_path = str(adapter_path)

        from unsloth import FastLanguageModel  # type: ignore[import-untyped]

        gguf_path = output_dir / f"model-{quantization}.gguf"
        FastLanguageModel.save_pretrained_gguf(
            model_path,
            model._tokenizer,
            str(gguf_path),
            quantization_method=quantization,
        )

        print(f"Model exported to GGUF: {gguf_path}")
        return str(gguf_path)

    except Exception as e:
        print(f"GGUF export error: {e}")
        raise


def push_to_huggingface(
    model: "BaseModel",
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    merge_adapter: bool = True,
) -> None:
    """Push model to Hugging Face Hub."""
    if not model._is_model_loaded:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    try:
        from huggingface_hub import HfApi, create_repo

        hf_token = token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "Hugging Face token required. Set HF_TOKEN env var or pass token parameter."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            assert model._fast_model is not None, "Fast model should be loaded"
            if merge_adapter:
                model._fast_model.save_pretrained(str(tmp_path), save_method="merged_16bit")
            else:
                model._fast_model.save_pretrained(str(tmp_path))

            model._tokenizer.save_pretrained(str(tmp_path))

            try:
                create_repo(repo_id=repo_id, token=hf_token, private=private, exist_ok=True)
            except Exception:
                pass

            api = HfApi()
            api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
            )

        print(f"Model pushed to Hugging Face: https://huggingface.co/{repo_id}")

    except ImportError as e:
        print(f"Hugging Face export error (missing dependency): {e}")
        print("Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        print(f"Hugging Face export error: {e}")
        raise


def export_to_ollama(
    model: "BaseModel",
    name: str,
    gguf_path: Optional[str] = None,
    template: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Export model to Ollama format."""
    try:
        if not gguf_path:
            gguf_path = export_gguf(model, output_path="./ollama_export")

        sys_prompt = system_prompt or ""
        modelfile_content = (
            f"FROM {gguf_path}\n\n"
            f'TEMPLATE """{{{{ if .System }}}}{sys_prompt}{{{{ end }}}}{{{{ .Prompt }}}}"""\n\n'
            f"PARAMETER temperature {model.inference_config.temperature}\n"
            f"PARAMETER top_p {model.inference_config.top_p}\n"
        )

        modelfile_path = Path(f"./{name}_Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        print(f"Modelfile created: {modelfile_path}")
        print(f"To create the Ollama model, run: ollama create {name} -f {modelfile_path}")

    except Exception as e:
        print(f"Ollama export error: {e}")
        raise


__all__ = [
    "add_expert",
    "remove_expert",
    "clear_experts",
    "set_expert_weights",
    "add_experts",
    "set_benchmark",
    "get_benchmark",
    "add_benchmark_sample",
    "load_benchmark",
    "save_checkpoint",
    "load_checkpoint",
    "restore_best_checkpoint",
    "list_checkpoints",
    "_call_model",
    "generate",
    "export_gguf",
    "push_to_huggingface",
    "export_to_ollama",
]
