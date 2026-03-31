"""Tool calling example.

This example demonstrates using built-in tools (python, web_search)
and creating custom tools for the model to use.

Run with: python examples/tool_calling.py
"""

from autotrain import Model
from autotrain.tools import python, web_search, create_tool, get_tool_schema

# Initialize model with tool calling support
model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")

# Add built-in tools
model.add_tool(python)
model.add_tool(web_search)


# Create a custom tool using the decorator
@create_tool(description="Calculate the factorial of a number")
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


model.add_tool(factorial)

# List available tools
print(f"Available tools: {model.list_tools()}")

# Generate with tool calling
# The model will automatically use tools when needed
result = model.generate_with_tools(
    prompt="Calculate 15 * 23 and then find the factorial of the result",
    temperature=0.7,
    max_tokens=500,
)

print(f"\nResult with tools:\n{result}")

# Get tool schema for documentation
print(f"\nTool schema for factorial:")
print(get_tool_schema(factorial))
