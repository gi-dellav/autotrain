"""Tool calling support for AutoTrain.

This module provides tool calling capabilities for both Model and Expert classes,
allowing LLMs to execute functions like Python code, calculators, and terminal commands.

Example usage:
    from autotrain import Model
    from autotrain.tools import python, calculator

    model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")
    model.add_tool(python)
    model.add_tool(calculator)

    result = model.generate("Calculate 15 * 23 and write fib(20)")
"""

import ast
import inspect
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class Tool:
    """
    Represents a callable tool that can be used by the LLM.

    Attributes:
        name: The name of the tool function
        description: Description of what the tool does (for the LLM)
        parameters: JSON schema for the tool parameters (OpenAI function format)
        function: The Python callable to execute
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool function."""
        return self.function(*args, **kwargs)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCallingConfig:
    """
    Manages a collection of tools for tool calling.

    Provides methods to add, remove, and manage tools.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def add_tool(self, tool: Union[Tool, Callable[..., Any]], name: Optional[str] = None) -> Tool:
        """
        Add a tool to the registry.

        Args:
            tool: A Tool instance or a callable function
            name: Optional name override (if callable is passed)

        Returns:
            The added Tool
        """
        if callable(tool) and not isinstance(tool, Tool):
            tool = create_tool()(tool)

        if name:
            tool.name = name

        self._tools[tool.name] = tool
        return tool

    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def clear_tools(self) -> None:
        """Remove all tools."""
        self._tools.clear()

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def get_tools(self) -> Dict[str, Tool]:
        """Get all tools as a dictionary."""
        return self._tools.copy()

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get all tools as OpenAI schemas."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with arguments.

        Args:
            name: Tool name
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found. Available tools: {self.list_tools()}")
        return tool(**arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def get_tool_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """
    Convert a Python function to OpenAI tool schema format.

    Args:
        func: A Python function

    Returns:
        OpenAI tool schema dictionary

    Example:
        def add(a: int, b: int) -> int:
            return a + b

        schema = get_tool_schema(add)
        # Returns:
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "add",
        #         "description": "Add two numbers",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "a": {"type": "integer", "description": "The first number"},
        #                 "b": {"type": "integer", "description": "The second number"}
        #             },
        #             "required": ["a", "b"]
        #         }
        #     }
        # }
    """
    return create_tool()(func).to_openai_schema()


def create_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    Decorator factory to create a Tool from a Python function.

    Args:
        name: Optional name override
        description: Optional description override
        parameters: Optional parameters schema override

    Returns:
        Decorator function

    Example:
        @create_tool(description="Get weather for a location")
        def get_weather(location: str, unit: str = "celsius") -> str:
            return f"Weather in {location}: 22{unit}"
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        _parameters = parameters
        if _parameters is None:
            _parameters = _extract_parameters(func)

        return Tool(
            name=tool_name,
            description=tool_description,
            parameters=_parameters,
            function=func,
        )

    return decorator


def _extract_parameters(func: Callable[..., Any]) -> Dict[str, Any]:
    """Extract parameters schema from a function signature."""
    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_info: Dict[str, Any] = {}

        if param.annotation is not inspect.Parameter.empty:
            param_type = param.annotation
            param_info["type"] = _python_type_to_json(param_type)

        if param.default is not inspect.Parameter.empty:
            param_info["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = param_info

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json(py_type: Any) -> str:
    """Convert Python type to JSON schema type."""
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    origin = getattr(py_type, "__origin__", None)
    if origin is Union:
        args = getattr(py_type, "__args__", ())
        if type(None) in args:
            non_none = [a for a in args if a is not None]
            if non_none:
                return _python_type_to_json(non_none[0])

    return type_map.get(py_type, "string")


# ==================== Built-in Tools ====================


def python(code: str) -> str:
    """
    Execute Python code and return the result.

    Args:
        code: Python code to execute

    Returns:
        String representation of the result

    Example:
        result = python("print('hello'); result = 2 + 2; result")
    """
    local_vars: dict[str, Any] = {}
    try:
        compiled = compile(code, "<string>", "exec")
        exec(compiled, {"__builtins__": __builtins__}, local_vars)

        if local_vars:
            result = {k: v for k, v in local_vars.items() if not k.startswith("_")}
            if result:
                if len(result) == 1:
                    return str(list(result.values())[0])
                return str(result)
        return "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"


# Disabled for security reason
'''
def terminal(command: str, working_dir: Optional[str] = None) -> str:
    """
    Execute a terminal command (with safety restrictions).

    Dangerous commands (rm, sudo, dd, chmod, etc.) are blocked.

    Args:
        command: Command to execute
        working_dir: Optional working directory

    Returns:
        Command output as string
    """
    dangerous_patterns = [
        r"\brm\\s+-rf\b",
        r"\brm\\s+-r\b",
        r"\bsudo\b",
        r"\bdd\b",
        r"\bchmod\\s+777\b",
        r"\bchmod\\s+-R\b",
        r"\bmkfs\b",
        r"\bfdisk\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b",
        r"\bpoweroff\b",
        r"\bmkdir\\s+-p\\s+/",
        r"\\:(){ :\\|:& };:",  # Fork bomb
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Error: Command blocked due to safety restrictions: {pattern}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            cwd=working_dir,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output or "Command executed successfully (no output)"
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
'''


def web_search(
    query: str, max_results: int = 5, model: str = "openrouter/google/gemini-3.0-flash-preview"
) -> str:
    """
    Search the web for information.

    Requires litellm to be installed.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        Search results as string
    """
    try:
        from litellm import completion

        response = completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Search the web for: {query}. Return the top {max_results} results with titles and brief descriptions.",
                }
            ],
            max_tokens=1000,
        )

        return response.choices[0].message.content or "No results found"

    except ImportError:
        return "Error: litellm is required for web search. Install with: pip install litellm"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"


# Create default tool registry with built-in tools
DEFAULT_TOOLS = ToolCallingConfig()
DEFAULT_TOOLS.add_tool(python)
DEFAULT_TOOLS.add_tool(web_search)


__all__ = [
    "Tool",
    "ToolCallingConfig",
    "create_tool",
    "get_tool_schema",
    "python",
    "web_search",
    "DEFAULT_TOOLS",
]
