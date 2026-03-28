"""Expert class for Autotrain - Teacher model based on litellm."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .config import InferenceConfig
    from .data_types import Sample, VisionSample
    from .tools import Tool

from .tools import ToolCallingConfig


@dataclass
class ExpertPrompts:
    """Configurable prompts for Expert components.

    For produce and solve, use the baking functions by calling get_produce() or get_solve().
    For select, review, and check, use the getter methods.

    If custom prompts are set, they will be used directly. Otherwise, the default detailed
    prompts from the prompts module will be used.

    Example:
        # Use default baked prompts
        prompts = ExpertPrompts()
        produce_prompt = prompts.get_produce("Python programming")

        # Use custom prompts
        custom_prompts = ExpertPrompts(
            produce="Custom produce prompt",
            solve="Custom solve prompt",
        )
    """

    produce: Optional[str] = None
    solve: Optional[str] = None
    select: Optional[str] = None
    review: Optional[str] = None
    check: Optional[str] = None

    def get_produce(self, topic: str, format: str = "json") -> str:
        """Get the produce prompt, using bake_producer if no custom prompt is set.

        Args:
            topic: The topic to generate training samples for.
            format: The desired output format (json, text, etc.).

        Returns:
            The produce prompt string.
        """
        from .prompts import bake_producer

        return self.produce if self.produce else bake_producer(topic, format)

    def get_solve(self, topic: Optional[str] = None, context: str = "") -> str:
        """Get the solve prompt, using bake_solver if no custom prompt is set.

        Args:
            topic: Optional topic context for the solver.
            context: Additional context or instructions.

        Returns:
            The solve prompt string.
        """
        from .prompts import bake_solver

        return self.solve if self.solve else bake_solver(topic, context)

    def get_select(self) -> str:
        """Get the select prompt using the default constant if no custom prompt is set.

        Returns:
            The select prompt string.
        """
        from .prompts import SPLITTER_DEFAULT

        return self.select if self.select else SPLITTER_DEFAULT

    def get_review(self) -> str:
        """Get the review prompt using the default constant if no custom prompt is set.

        Returns:
            The review prompt string.
        """
        from .prompts import REVIEWER_DEFAULT

        return self.review if self.review else REVIEWER_DEFAULT

    def get_check(self) -> str:
        """Get the check prompt using the default constant if no custom prompt is set.

        Returns:
            The check prompt string.
        """
        from .prompts import CHECKER_DEFAULT

        return self.check if self.check else CHECKER_DEFAULT


class Expert:
    """
    Expert class wrapping a litellm model, acting as a teacher.

    The expert can generate samples, solve problems, and provide guidance
    to the fine-tuned model during training.

    Supports multiple experts with weighted production rates in the Model class.
    Example usage:
        expert1 = Expert(model_name="gpt-4", production_rate=0.5)
        expert2 = Expert(model_name="claude-3", production_rate=0.25)
        model.train(experts=[(expert1, 0.5), (expert2, 0.25)])
        # This means: 50% from expert1, 25% from expert2, 25% from model
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.5-397b-a17b",
        production_rate: float = 0.5,
        inference_config: Optional["InferenceConfig"] = None,
        prompts: Optional[ExpertPrompts] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_llm_provider: str = "openrouter",
        avoid_solving_same_sample: bool = False,
        avoid_checking_same_sample: bool = False,
        thinking: bool = True,
    ):
        """
        Initialize the Expert.

        Args:
            model_name: litellm model name (e.g., "gpt-4", "claude-3", "ollama/llama2") (default: qwen/qwen3.5-397b-a17b)
            production_rate: Rate at which expert produces samples vs student model
                            (0.5 = 50% expert, 50% student when used alone)
            inference_config: Configuration for inference properties
            prompts: Custom prompts for expert operations
            api_key: Optional API key for the model provider
            api_base: Optional API base URL
            api_version: Optional API version (for Azure)
            custom_llm_provider: Optional custom LLM provider (default: openrouter)
            avoid_solving_same_sample: If True, skip solving samples produced by this expert
            avoid_checking_same_sample: If True, skip checking samples solved by this expert
            thinking: Whether to enable thinking/reasoning (default: True)
        """
        self.model_name = model_name
        self.production_rate = production_rate
        self.inference_config = inference_config or self._default_inference_config()
        self.prompts = prompts or ExpertPrompts()
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.custom_llm_provider = custom_llm_provider
        self.avoid_solving_same_sample = avoid_solving_same_sample
        self.avoid_checking_same_sample = avoid_checking_same_sample
        self.thinking = thinking

        # Dataset storage
        self._samples: list["Sample"] = []
        self._training_data: list[dict] = []

        # Caching
        self._cache_enabled = False
        self._cache: dict[str, str] = {}

        # Tool calling
        self._tools = ToolCallingConfig()

    def _default_inference_config(self):
        """Create a default InferenceConfig."""
        from .config import InferenceConfig

        return InferenceConfig()

    def _call_litellm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Make a litellm API call.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            Model response
        """
        # Check cache
        cache_key = f"{prompt}:{system_prompt}"
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from litellm import completion

            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=(
                    temperature if temperature is not None else self.inference_config.temperature
                ),
                max_tokens=(
                    max_tokens if max_tokens is not None else self.inference_config.max_tokens
                ),
                top_p=self.inference_config.top_p,
                frequency_penalty=self.inference_config.frequency_penalty,
                presence_penalty=self.inference_config.presence_penalty,
                api_key=self.api_key,
                api_base=self.api_base,
                api_version=self.api_version,
                custom_llm_provider=self.custom_llm_provider,
                reasoning_effort="high"
                if self.thinking and self.inference_config.thinking
                else None,
            )

            result = response.choices[0].message.content

            # Cache result
            if self._cache_enabled:
                self._cache[cache_key] = result

            return result

        except ImportError:
            raise ImportError("litellm is required. Install with: pip install litellm")
        except Exception as e:
            raise RuntimeError(f"litellm API call failed: {e}")

    def enable_cache(self) -> None:
        """Enable response caching."""
        self._cache_enabled = True
        self._cache.clear()

    def disable_cache(self) -> None:
        """Disable response caching."""
        self._cache_enabled = False

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()

    def produce(self, task: str, count: int = 1) -> list["Sample"]:
        """
        Produce training samples as an expert.

        Args:
            task: The task/domain to generate samples for
            count: Number of samples to generate

        Returns:
            List of generated samples
        """

        samples = []

        for _ in range(count):
            prompt = self.prompts.get_produce(task)
            prompt += "\n\nGenerate one high-quality input-output pair."
            response = self._call_litellm(prompt=prompt)

            # Parse the response to extract input/output
            sample = self._parse_sample_response(response)
            if sample:
                sample.metadata["producer"] = f"expert:{self.model_name}"
                samples.append(sample)

        self._samples.extend(samples)
        return samples

    def solve(
        self,
        input_data: str,
        metadata: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Solve a problem as an expert.

        Args:
            input_data: The problem/input to solve
            metadata: Optional sample metadata to check for self-solving
            system_prompt: Optional system prompt override
            temperature: Optional temperature override

        Returns:
            The expert's solution
        """
        if self.avoid_solving_same_sample and metadata is not None:
            producer_key = f"expert:{self.model_name}"
            if metadata.get("producer") == producer_key:
                return ""

        prompt = self.prompts.get_solve()
        prompt += f"\n\n{input_data}"
        return self._call_litellm(
            prompt=prompt,
            system_prompt=system_prompt
            or "You are an expert assistant. Provide clear, accurate solutions.",
            temperature=temperature,
        )

    def solve_batch(
        self,
        inputs: list[str],
        system_prompt: Optional[str] = None,
    ) -> list[str]:
        """
        Solve multiple problems.

        Args:
            inputs: List of inputs to solve
            system_prompt: Optional system prompt

        Returns:
            List of solutions
        """
        return [self.solve(inp, system_prompt) for inp in inputs]

    def select(self, samples: list[str]) -> str:
        """
        Select the most useful sample from options.

        Args:
            samples: List of sample outputs to choose from

        Returns:
            The selected sample
        """
        samples_text = "\n\n".join([f"Option {i + 1}:\n{s}" for i, s in enumerate(samples)])

        prompt = self.prompts.get_select()
        prompt += f"\n\n{samples_text}\n\n"
        prompt += "Respond with only the option number (1, 2, 3, etc.) of the best sample."

        response = self._call_litellm(prompt=prompt)

        # Parse the selected option
        match = re.search(r"\d+", response)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(samples):
                return samples[idx]

        return samples[0] if samples else ""

    def review(
        self,
        sample: "Sample",
        criteria: Optional[str] = None,
    ) -> dict:
        """
        Review a sample and provide feedback.

        Args:
            sample: The sample to review
            criteria: Optional custom review criteria

        Returns:
            Dictionary with review feedback
        """
        criteria_prompt = ""
        if criteria:
            criteria_prompt = f"\nReview criteria: {criteria}"

        prompt = self.prompts.get_review()
        prompt += f"\n\nInput: {sample.input_data}\n\nOutput: {sample.output_data}"
        if criteria:
            prompt += f"\n\nReview criteria: {criteria}"
        prompt += "\n\nProvide a quality score (1-10) and brief feedback."
        response = self._call_litellm(
            prompt=prompt,
            system_prompt="You are an expert reviewer. Provide constructive feedback.",
        )

        # Parse score and feedback
        score_match = re.search(r"(?:score|rating)[:\s]*(\d+)", response.lower())
        score = int(score_match.group(1)) if score_match else 5

        return {
            "score": score,
            "feedback": response,
            "reviewer": "expert",
            "expert_model": self.model_name,
        }

    def check(
        self,
        input_data: str,
        output_data: str,
        metadata: Optional[dict] = None,
        strict: bool = False,
    ) -> dict:
        """
        Verify the correctness of a solution.

        Args:
            input_data: The original problem/input
            output_data: The proposed solution
            metadata: Optional sample metadata to check for self-checking
            strict: If True, require exact correctness

        Returns:
            Dictionary with verification results
        """
        if self.avoid_checking_same_sample and metadata is not None:
            solver_key = f"expert:{self.model_name}"
            if metadata.get("solver") == solver_key:
                return {
                    "is_correct": None,
                    "explanation": "Skipped self-check: expert cannot verify its own solution",
                    "checker": "expert",
                    "expert_model": self.model_name,
                    "strict": strict,
                    "skipped": True,
                }

        strictness = (
            "Be very strict in your evaluation." if strict else "Be reasonable in your evaluation."
        )

        prompt = self.prompts.get_check()
        prompt += f"\n\nProblem: {input_data}\n\nSolution: {output_data}\n\n"
        prompt += f"{strictness}\n"
        prompt += "Is this solution correct? Respond with YES or NO, then explain."
        response = self._call_litellm(
            prompt=prompt,
            system_prompt=f"You are an expert verifier. Be precise and accurate. {strictness}",
        )

        is_correct = "YES" in response.upper().split("\n")[0]

        return {
            "is_correct": is_correct,
            "explanation": response,
            "checker": "expert",
            "expert_model": self.model_name,
            "strict": strict,
        }

    def compare(
        self,
        input_data: str,
        output_a: str,
        output_b: str,
    ) -> dict:
        """
        Compare two outputs and determine which is better.

        Args:
            input_data: The original input
            output_a: First output
            output_b: Second output

        Returns:
            Dictionary with comparison results
        """
        response = self._call_litellm(
            prompt=(
                f"Compare the following two responses to the same input.\n\n"
                f"Input: {input_data}\n\n"
                f"Response A:\n{output_a}\n\n"
                f"Response B:\n{output_b}\n\n"
                "Which response is better? Consider accuracy, completeness, "
                "clarity, and helpfulness.\n"
                "Respond with:\n"
                '- "A" if Response A is better\n'
                '- "B" if Response B is better\n'
                '- "TIE" if they are equally good\n\n'
                "Then explain your reasoning."
            ),
            system_prompt="You are an expert evaluator comparing response quality.",
        )

        # Parse the winner
        response_upper = response.upper()
        if "TIE" in response_upper:
            winner = "tie"
        elif "\nA" in response_upper or response_upper.startswith("A"):
            winner = "a"
        else:
            winner = "b"

        return {"winner": winner, "explanation": response, "expert_model": self.model_name}

    def rate(
        self,
        input_data: str,
        output_data: str,
        scale: int = 10,
        criteria: Optional[list[str]] = None,
    ) -> dict:
        """
        Rate an output on a numerical scale.

        Args:
            input_data: The input
            output_data: The output to rate
            scale: Maximum score (default 10)
            criteria: Optional list of criteria to rate

        Returns:
            Dictionary with ratings
        """
        criteria_prompt = ""
        if criteria:
            criteria_prompt = "\nRate based on these criteria:\n" + "\n".join(
                f"- {c}" for c in criteria
            )

        response = self._call_litellm(
            prompt=f"""Rate the following response on a scale of 1 to {scale}.{criteria_prompt}

Input: {input_data}
Output: {output_data}

Provide an overall score and optionally scores for each criterion.""",
            system_prompt="You are an expert evaluator.",
        )

        # Extract overall score
        score_match = re.search(r"(?:score|rating|overall)[:\s]*(\d+)", response.lower())
        score = int(score_match.group(1)) if score_match else scale // 2

        return {
            "score": score,
            "max_score": scale,
            "normalized_score": score / scale,
            "details": response,
            "expert_model": self.model_name,
        }

    def _parse_sample_response(self, response: str) -> Optional["Sample"]:
        """Parse a sample from expert response."""
        # Import here to avoid circular dependency
        # Try to extract input/output from various formats
        import re

        from .data_types import Sample

        # Try JSON format first
        json_match = re.search(r'\{[^}]*"input"[^}]*"output"[^}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return Sample(
                    input_data=data.get("input", ""),
                    output_data=data.get("output", ""),
                    metadata={"source": "expert"},
                )
            except json.JSONDecodeError as e:
                # JSON parsing failed, try other formats
                print(f"Warning: JSON decode failed in expert response: {e}")

        # Try "Input:" / "Output:" format
        input_match = re.search(
            r"Input[:\s]*(.+?)(?=Output|$)", response, re.DOTALL | re.IGNORECASE
        )
        output_match = re.search(r"Output[:\s]*(.+?)$", response, re.DOTALL | re.IGNORECASE)

        if input_match and output_match:
            return Sample(
                input_data=input_match.group(1).strip(),
                output_data=output_match.group(1).strip(),
                metadata={"source": "expert"},
            )

        # Fallback: treat entire response as output with empty input
        return Sample(input_data="", output_data=response, metadata={"source": "expert"})

    def add_sample(self, input_data: str, output_data: str, metadata: Optional[dict] = None):
        """Add a sample to the expert's dataset."""
        from .data_types import Sample

        sample = Sample(
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {"source": "expert"},
        )
        self._samples.append(sample)

    def export_dataset(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Export the expert's dataset to a file.

        Args:
            path: Output file path
            format: Export format ("json" or "jsonl")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {"input": s.input_data, "output": s.output_data, "metadata": s.metadata}
            for s in self._samples
        ]

        if format == "jsonl":
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        print(f"Expert dataset exported to {path}")

    def import_dataset(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Import a dataset into the expert.

        Args:
            path: Input file path
            format: Import format ("json" or "jsonl")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if format == "jsonl":
            with open(path) as f:
                data = [json.loads(line) for line in f]
        else:
            with open(path) as f:
                data = json.load(f)

        for item in data:
            self.add_sample(
                input_data=item.get("input", ""),
                output_data=item.get("output", ""),
                metadata=item.get("metadata", {}),
            )

        print(f"Imported {len(data)} samples to expert from {path}")

    def get_samples(self) -> list["Sample"]:
        """Get all samples."""
        return self._samples.copy()

    def clear_samples(self) -> None:
        """Clear all samples."""
        self._samples.clear()
        self._training_data.clear()

    def set_production_rate(self, rate: float) -> None:
        """
        Set the expert's production rate.

        Args:
            rate: Production rate (0.0 to 1.0)
                  Higher = more samples from expert, lower = more from student
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Production rate must be between 0.0 and 1.0")
        self.production_rate = rate

    def set_inference_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> None:
        """
        Update inference configuration.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            top_p: Top-p sampling
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
        """
        if temperature is not None:
            self.inference_config.temperature = temperature
        if max_tokens is not None:
            self.inference_config.max_tokens = max_tokens
        if top_p is not None:
            self.inference_config.top_p = top_p
        if frequency_penalty is not None:
            self.inference_config.frequency_penalty = frequency_penalty
        if presence_penalty is not None:
            self.inference_config.presence_penalty = presence_penalty

    def get_info(self) -> dict:
        """Get expert information."""
        return {
            "model_name": self.model_name,
            "production_rate": self.production_rate,
            "cache_enabled": self._cache_enabled,
            "samples_count": len(self._samples),
            "inference_config": {
                "temperature": self.inference_config.temperature,
                "max_tokens": self.inference_config.max_tokens,
                "top_p": self.inference_config.top_p,
            },
            "tools": self.list_tools(),
        }

    # ==================== Tool Management ====================

    def add_tool(
        self,
        tool: Union["Tool", Callable[..., Any]],
        name: Optional[str] = None,
    ) -> "Tool":
        """
        Add a tool for tool calling.

        Args:
            tool: A Tool instance or a callable function
            name: Optional name override

        Returns:
            The added Tool

        Example:
            from autotrain.tools import python, calculator

            expert.add_tool(python)
            expert.add_tool(calculator)
        """
        return self._tools.add_tool(tool, name)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        return self._tools.remove_tool(name)

    def clear_tools(self) -> None:
        """Remove all tools."""
        self._tools.clear_tools()

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return self._tools.list_tools()

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return self._tools.has_tool(name)

    def get_tool(self, name: str) -> Optional["Tool"]:
        """Get a tool by name."""
        return self._tools.get_tool(name)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tools as OpenAI schemas."""
        return self._tools.get_schemas()

    def create_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], "Tool"]:
        """
        Decorator to create a tool from a function.

        Args:
            name: Optional name override
            description: Optional description override

        Returns:
            Decorator function
        """
        from .tools import create_tool as _create_tool

        return _create_tool(name=name, description=description)

    # ==================== Tool Calling Methods ====================

    def _call_litellm_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Call litellm with tool calling support.

        Args:
            messages: List of message dicts
            tools: Optional list of tool schemas

        Returns:
            Response with tool_calls if any
        """
        try:
            from litellm import completion
        except ImportError:
            raise ImportError("litellm is required. Install with: pip install litellm")

        tool_schemas = tools or self._tools.get_schemas()

        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=self.inference_config.temperature,
                max_tokens=self.inference_config.max_tokens,
                top_p=self.inference_config.top_p,
                tools=tool_schemas if tool_schemas else None,
                api_key=self.api_key,
                api_base=self.api_base,
                api_version=self.api_version,
                custom_llm_provider=self.custom_llm_provider,
                reasoning_effort="high"
                if self.thinking and self.inference_config.thinking
                else None,
            )

            result = response.choices[0].message

            return {
                "content": result.content or "",
                "tool_calls": getattr(result, "tool_calls", None) or [],
            }

        except Exception as e:
            return {"content": f"[Error: {e}]", "tool_calls": []}

    def solve_with_tools(
        self,
        input_data: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tool_calls: int = 10,
    ) -> str:
        """
        Solve a problem with tool calling support.

        Args:
            input_data: The problem/input to solve
            tools: Optional list of tool schemas (uses registered tools if None)
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tool_calls: Maximum tool call iterations

        Returns:
            Solution with tool execution results
        """
        sys_prompt = (
            system_prompt or "You are an expert assistant. Provide clear, accurate solutions."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{self.prompts.solve}\n\n{input_data}"},
        ]

        tool_schemas = tools or self._tools.get_schemas()

        if not tool_schemas:
            return self.solve(input_data, system_prompt=system_prompt, temperature=temperature)

        tool_call_count = 0
        final_content = ""

        while tool_call_count < max_tool_calls:
            response = self._call_litellm_with_tools(messages, tool_schemas)

            content = response.get("content", "")
            final_content = content

            messages.append({"role": "assistant", "content": content})

            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {"code": tool_call.function.arguments}

                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    }
                )

                try:
                    tool_result = self._tools.execute_tool(tool_name, tool_args)
                    result_str = str(tool_result)
                except Exception as e:
                    result_str = f"Error executing tool '{tool_name}': {str(e)}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

                tool_call_count += 1

            if tool_call_count >= max_tool_calls:
                final_content += f"\n\n[Max tool calls ({max_tool_calls}) reached]"
                break

        return final_content

    def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_tool_calls: int = 10,
    ) -> str:
        """
        Generate text with tool calling support.

        Args:
            prompt: User prompt
            tools: Optional list of tool schemas
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            max_tool_calls: Maximum tool call iterations

        Returns:
            Generated text with tool execution results
        """
        sys_prompt = system_prompt or "You are a helpful assistant."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        tool_schemas = tools or self._tools.get_schemas()

        if not tool_schemas:
            if temperature:
                self.inference_config.temperature = temperature
            if max_tokens:
                self.inference_config.max_tokens = max_tokens
            return self._call_litellm(
                prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
            )

        tool_call_count = 0
        final_content = ""

        temp = temperature or self.inference_config.temperature
        maxtok = max_tokens or self.inference_config.max_tokens

        while tool_call_count < max_tool_calls:
            try:
                from litellm import completion

                response = completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=maxtok,
                    top_p=self.inference_config.top_p,
                    tools=tool_schemas,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    custom_llm_provider=self.custom_llm_provider,
                    reasoning_effort="high"
                    if self.thinking and self.inference_config.thinking
                    else None,
                )

                result = response.choices[0].message
                content = result.content or ""
                final_content = content

                messages.append({"role": "assistant", "content": content})

                tool_calls = getattr(result, "tool_calls", None) or []

                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {"code": tool_call.function.arguments}

                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                            ],
                        }
                    )

                    try:
                        tool_result = self._tools.execute_tool(tool_name, tool_args)
                        result_str = str(tool_result)
                    except Exception as e:
                        result_str = f"Error executing tool '{tool_name}': {str(e)}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )

                    tool_call_count += 1

                if tool_call_count >= max_tool_calls:
                    final_content += f"\n\n[Max tool calls ({max_tool_calls}) reached]"
                    break

            except Exception as e:
                final_content = f"[Error: {str(e)}]"
                break

        return final_content


class VisionExpert(Expert):
    """
    Vision Expert class for VLM-based experts.

    Extends Expert to support vision-language models for multi-modal tasks.
    Uses litellm with OpenRouter as default provider.

    Example usage:
        expert = VisionExpert(model_name="qwen/qwen2.5-vl-7b-instruct")
        model.train(experts=[(expert, 0.5)])
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.5-397b-a17b",
        production_rate: float = 0.5,
        inference_config: Optional["InferenceConfig"] = None,
        prompts: Optional[ExpertPrompts] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_llm_provider: str = "openrouter",
        avoid_solving_same_sample: bool = False,
        avoid_checking_same_sample: bool = False,
        thinking: bool = True,
    ):
        """
        Initialize the Vision Expert.

        Args:
            model_name: litellm model name for VLM (default: qwen/qwen3.5-397b-a17b)
            production_rate: Rate at which expert produces samples vs student model
            inference_config: Configuration for inference properties
            prompts: Custom prompts for expert operations
            api_key: Optional API key for the model provider
            api_base: Optional API base URL
            api_version: Optional API version (for Azure)
            custom_llm_provider: Optional custom LLM provider (default: openrouter)
            avoid_solving_same_sample: If True, skip solving samples produced by this expert
            avoid_checking_same_sample: If True, skip checking samples solved by this expert
            thinking: Whether to enable thinking/reasoning (default: True)
        """
        super().__init__(
            model_name=model_name,
            production_rate=production_rate,
            inference_config=inference_config,
            prompts=prompts,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            custom_llm_provider=custom_llm_provider,
            avoid_solving_same_sample=avoid_solving_same_sample,
            avoid_checking_same_sample=avoid_checking_same_sample,
            thinking=thinking,
        )
        self.vision_model = True
        self._vision_samples: list = []

    def _call_litellm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list] = None,
        **kwargs,
    ) -> str:
        """
        Call litellm with optional image support.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            images: Optional list of images to include

        Returns:
            Generated text response
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("litellm is required. Install with: pip install litellm")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if images:
            for img in images:
                if isinstance(img, str):
                    user_content.append({"type": "image_url", "image_url": {"url": img}})
                else:
                    user_content.append({"type": "image", "image": img})

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        temperature = temperature or self.inference_config.temperature
        max_tokens = max_tokens or self.inference_config.max_tokens

        response = litellm.completion(
            model=f"{self.custom_llm_provider}/{self.model_name}"
            if self.custom_llm_provider
            else self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            base_url=self.api_base,
            api_version=self.api_version,
            reasoning_effort="high" if self.thinking and self.inference_config.thinking else None,
            **kwargs,
        )

        return response.choices[0].message.content

    def solve(
        self,
        input_data: str,
        images: Optional[list] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Solve a problem using the vision model.

        Args:
            input_data: The input problem/task
            images: Optional images for vision tasks
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Returns:
            Generated solution
        """
        prompt = f"{self.prompts.solve} {input_data}"
        cache_key = f"solve:{input_data}:{str(images)}"

        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        result = self._call_litellm(
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            **kwargs,
        )

        if self._cache_enabled:
            self._cache[cache_key] = result

        return result

    def solve_batch(
        self,
        inputs: list,
        images: Optional[list] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> list:
        """
        Solve multiple problems in batch.

        Args:
            inputs: List of input problems
            images: Optional list of images (one per input)
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Returns:
            List of solutions
        """
        results = []
        for i, input_data in enumerate(input_data):
            input_images = images[i] if images and i < len(images) else None
            result = self.solve(input_data, input_images, system_prompt, **kwargs)
            results.append(result)
        return results

    def produce(
        self,
        task_description: str,
        count: int = 1,
        **kwargs,
    ) -> list:
        """
        Generate training samples for a vision task.

        Args:
            task_description: Description of the task
            count: Number of samples to generate
            **kwargs: Additional arguments

        Returns:
            List of generated samples
        """
        from .data_types import VisionSample

        samples = []
        for i in range(count):
            prompt = f"{self.prompts.produce} {task_description}. Generate sample {i + 1}."

            result = self._call_litellm(
                prompt=prompt,
                **kwargs,
            )

            sample = VisionSample(
                input_data=task_description,
                output_data=result,
                metadata={"source": "expert", "expert": self.model_name, "index": i},
            )
            samples.append(sample)

        self._samples.extend(samples)
        return samples

    def add_sample(self, sample: "VisionSample") -> None:
        """Add a vision sample to the expert's dataset."""
        self._samples.append(sample)
        if hasattr(sample, "images"):
            self._vision_samples.append(sample)

    def get_vision_samples(self) -> list:
        """Get all vision samples."""
        return self._vision_samples

    def clear_vision_samples(self) -> None:
        """Clear all vision samples."""
        self._vision_samples = []

    def get_info(self) -> dict:
        """Get vision expert information."""
        info = super().get_info()
        info["vision_model"] = self.vision_model
        info["vision_samples_count"] = len(self._vision_samples)
        return info
