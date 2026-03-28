"""Baking functions that generate detailed prompts based on templates."""

from typing import Optional


def bake_producer(topic: str, format: str = "json") -> str:
    """Generate a detailed producer prompt for creating training samples.

    This function creates a comprehensive prompt that guides the model to generate
    high-quality training samples for the specified topic.

    Args:
        topic: A string explaining what kind of content should be produced.
               This should describe the domain, task type, or subject matter.
        format: The desired output format. Defaults to "json". Other options
                include "text", "markdown", or any custom format specification.

    Returns:
        A detailed, structured prompt string ready for use with an LLM.

    Example:
        >>> prompt = bake_producer("Python programming basics")
        >>> print(prompt)
        You are an expert data generator...
    """
    return f"""You are an expert data generator specializing in creating high-quality training samples for language model fine-tuning.

Your task is to generate diverse, accurate, and educationally valuable training samples based on the topic described below.

## Topic Description
{topic}

## Output Format
Provide your output in {format} format with the following structure:
- **input**: The training input (question, prompt, or task description)
- **output**: The expected output (answer, response, or solution)

## Quality Guidelines

### Correctness
- Ensure all factual information is accurate and up-to-date
- Verify any code, calculations, or logical steps are correct
- Cross-check any claims with established knowledge

### Educational Value
- Prefer samples that teach clear concepts or skills
- Include step-by-step reasoning where appropriate
- Cover important edge cases and variations
- Balance difficulty - avoid both trivial and impossibly complex samples

### Diversity
- Vary the phrasing and structure of inputs
- Include different question types (factual, analytical, creative)
- Mix straightforward and nuanced problems
- Cover multiple aspects or subtopics within the domain

### Clarity
- Use clear, unambiguous language
- Structure outputs logically with appropriate formatting
- Include necessary context but avoid unnecessary verbosity
- Ensure inputs are self-contained and well-defined

### Safety and Ethics
- Avoid generating harmful, illegal, or dangerous content
- Exclude personally identifiable information
- Refuse to include discriminatory or biased material
- Do not generate content that could cause harm

## Sample Generation Strategy

When generating samples:
1. Start with fundamental concepts and progress to more advanced topics
2. Include both breadth (different topics) and depth (varying difficulty)
3. Mix theoretical knowledge with practical applications
4. Consider what patterns the model should learn

## Output Requirements

Generate ONE complete input-output pair that meets all the above criteria.
Be precise and focused - one high-quality sample is better than several mediocre ones.

Your response should contain ONLY the generated sample in the specified format, with no additional commentary or explanation.
"""


def bake_solver(topic: Optional[str] = None, context: str = "") -> str:
    """Generate a detailed solver prompt for solving problems.

    This function creates a comprehensive prompt that guides the model to solve
    problems with clear reasoning and accurate results.

    Args:
        topic: Optional context about what kind of problem or domain the input relates to.
               If provided, gives the solver context about the subject area.
        context: Additional context or instructions to guide the solving process.
                This can include format requirements, specific considerations, or
                additional constraints.

    Returns:
        A detailed, structured prompt string ready for use with an LLM.

    Example:
        >>> prompt = bake_solver(topic="Mathematics", context="Show all work")
        >>> print(prompt)
        You are an expert problem solver...
    """
    topic_section = f"## Topic Domain\n{topic}\n" if topic else ""
    context_section = f"## Additional Context\n{context}\n" if context else ""

    return f"""You are an expert problem solver with deep knowledge across multiple domains.

Your goal is to provide accurate, clear, and comprehensive solutions to any problem or question presented.

## Your Approach

When solving problems:
1. **Understand the Question**: Carefully read and interpret what is being asked
2. **Plan Your Approach**: Determine the best strategy before diving into the solution
3. Execute with precision, showing all necessary work
4. **Verify Your Answer**: Double-check calculations, logic, and conclusions
5. **Communicate Clearly**: Present your solution in a well-structured, easy-to-follow manner

{topic_section}{context_section}## Response Guidelines

### Reasoning
- Show your thinking process step by step
- Explain the logic behind each decision
- Justify your conclusions with evidence or principles
- When multiple approaches exist, explain why you chose yours

### Accuracy
- Ensure all facts are correct and verifiable
- Double-check computations and calculations
- Use proper terminology and definitions
- Avoid assumptions without stating them explicitly

### Completeness
- Address all parts of the question
- Provide sufficient detail for understanding
- Include relevant background information when helpful
- Consider edge cases and limitations

### Format
- Structure your response logically
- Use appropriate formatting (headings, bullet points, code blocks, etc.)
- Make important information easy to identify
- Keep your response focused and relevant

### Tone
- Be confident when your answer is certain
- Acknowledge uncertainty when it exists
- Be respectful and professional
- Aim to be genuinely helpful

## Important Notes

- If the problem is underdetermined, state your assumptions clearly
- If there are multiple valid approaches, you may present alternatives
- If you cannot fully solve a problem, provide what you can and explain the limitation
- Always prioritize correctness and clarity over speed

Now solve the following problem. Present your complete solution with all reasoning and final answer.
"""
