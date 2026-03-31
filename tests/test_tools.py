"""Tests for tools module."""

import pytest
from autotrain.tools import (
    Tool,
    ToolCallingConfig,
    create_tool,
    get_tool_schema,
    python,
    web_search,
    DEFAULT_TOOLS,
)


class TestToolDataclass:
    """Tests for Tool dataclass."""

    def test_tool_creation(self):
        """Test creating a Tool manually."""

        def dummy_func(x: int) -> int:
            return x * 2

        tool = Tool(
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            function=dummy_func,
        )

        assert tool.name == "double"
        assert tool.description == "Double a number"
        assert tool(5) == 10

    def test_tool_to_openai_schema(self):
        """Test converting Tool to OpenAI schema."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = create_tool(description="Add two numbers together")(add)

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "add"
        assert "description" in schema["function"]


class TestToolCallingConfig:
    """Tests for ToolCallingConfig class."""

    def test_default_tools_not_empty(self):
        """Test DEFAULT_TOOLS has built-in tools."""
        assert len(DEFAULT_TOOLS) > 0
        assert DEFAULT_TOOLS.has_tool("python")
        assert DEFAULT_TOOLS.has_tool("web_search")

    def test_add_tool_from_callable(self):
        """Test adding a tool from a callable."""

        def my_func(x: int) -> int:
            return x

        config = ToolCallingConfig()
        tool = config.add_tool(my_func)

        assert config.has_tool("my_func")
        assert tool.name == "my_func"

    def test_add_tool_with_name_override(self):
        """Test adding tool with name override."""

        def my_func(x: int) -> int:
            return x

        config = ToolCallingConfig()
        config.add_tool(my_func, name="custom_name")

        assert config.has_tool("custom_name")
        assert not config.has_tool("my_func")

    def test_remove_tool(self):
        """Test removing a tool."""
        config = ToolCallingConfig()
        config.add_tool(python)

        assert config.has_tool("python")
        assert config.remove_tool("python")
        assert not config.has_tool("python")
        assert not config.remove_tool("python")

    def test_clear_tools(self):
        """Test clearing all tools."""
        config = ToolCallingConfig()
        config.add_tool(python)
        config.add_tool(web_search)

        assert len(config) == 2
        config.clear_tools()
        assert len(config) == 0

    def test_list_tools(self):
        """Test listing tools."""
        config = ToolCallingConfig()
        config.add_tool(python)
        config.add_tool(web_search)

        tools = config.list_tools()
        assert "python" in tools
        assert "web_search" in tools

    def test_get_tool(self):
        """Test getting a tool."""
        config = ToolCallingConfig()
        config.add_tool(python)

        tool = config.get_tool("python")
        assert tool is not None
        assert tool.name == "python"

    def test_get_schemas(self):
        """Test getting OpenAI schemas."""
        config = ToolCallingConfig()
        config.add_tool(python)
        config.add_tool(web_search)

        schemas = config.get_schemas()

        assert len(schemas) == 2
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema

    def test_execute_tool(self):
        """Test executing a tool."""
        config = ToolCallingConfig()
        config.add_tool(python)

        result = config.execute_tool("python", {"code": "x = 15 * 23; x"})
        assert "345" in result

    def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool raises error."""
        config = ToolCallingConfig()

        with pytest.raises(ValueError, match="not found"):
            config.execute_tool("nonexistent", {})


class TestBuiltinTools:
    """Tests for built-in tools."""

    def test_python_tool_basic(self):
        """Test Python tool basic execution."""
        result = python("x = 2 + 2; x")
        assert "4" in result

    def test_python_tool_variable(self):
        """Test Python tool with variable."""
        result = python("x = 5; y = 10; result = x + y")
        assert "15" in result

    def test_python_tool_error(self):
        """Test Python tool error handling."""
        result = python("1/0")
        assert "Error" in result

    def test_web_search(self):
        """Test web_search tool."""
        result = web_search("Python programming")
        # web_search may fail or return results depending on API availability
        assert isinstance(result, str)


class TestCreateTool:
    """Tests for create_tool decorator."""

    def test_create_tool_from_function(self):
        """Test creating a tool from a function."""

        @create_tool(description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        tool = multiply
        assert tool.name == "multiply"
        assert "Multiply two numbers" in tool.description
        assert tool(3, 4) == 12

    def test_create_tool_with_name_override(self):
        """Test creating tool with name override."""

        @create_tool(name="custom_multiply")
        def orig_name(a: int, b: int) -> int:
            return a * b

        assert orig_name.name == "custom_multiply"

    def test_create_tool_with_default_value(self):
        """Test creating tool with default parameter."""

        @create_tool(description="Greet someone")
        def greet(name: str = "World") -> str:
            return f"Hello, {name}!"

        assert greet() == "Hello, World!"
        assert greet("Alice") == "Hello, Alice!"

    def test_create_tool_schema(self):
        """Test tool schema extraction."""

        @create_tool(description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        schema = add.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"] == "Add numbers"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]


class TestGetToolSchema:
    """Tests for get_tool_schema function."""

    def test_get_tool_schema(self):
        """Test get_tool_schema function."""

        def calculate(a: int, b: int, operation: str = "add") -> int:
            """Perform calculation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        schema = get_tool_schema(calculate)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculate"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]
        assert "operation" in schema["function"]["parameters"]["properties"]


class TestToolIntegration:
    """Integration tests for tools."""

    def test_full_tool_workflow(self):
        """Test full tool workflow."""
        config = ToolCallingConfig()

        @create_tool(description="Calculate square")
        def square(n: int) -> int:
            return n * n

        config.add_tool(square)
        assert config.has_tool("square")

        schema = config.get_schemas()
        assert len(schema) == 1

        result = config.execute_tool("square", {"n": 5})
        assert result == 25

    def test_multiple_tools(self):
        """Test using multiple tools."""
        config = ToolCallingConfig()
        config.add_tool(python)
        config.add_tool(web_search)

        assert len(config) == 2

        py_result = config.execute_tool("python", {"code": "x = 10 + 5; x"})
        assert "15" in py_result

        search_result = config.execute_tool("web_search", {"query": "test"})
        assert isinstance(search_result, str)
