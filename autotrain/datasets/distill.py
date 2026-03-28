"""Expert distillation utilities for AutoTrain."""

from typing import Any, Dict, List, Optional, Union

from autotrain.data_types import Sample
from autotrain.expert import Expert


class ExpertDistiller:
    """Expert distillation utilities.

    Uses an Expert model to generate training data through knowledge distillation.
    The expert model (teacher) generates high-quality samples that can be used
    to fine-tune a smaller model (student).

    Supports:
    - Direct generation from task descriptions
    - Generation from existing prompts/inputs
    - Format conversion for training
    """

    def __init__(
        self,
        expert: Optional[Expert] = None,
        model_name: str = "deepseek/deepseek-v3.2",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """Initialize ExpertDistiller.

        Args:
            expert: Pre-configured Expert instance. If None, creates one.
            model_name: Model name for new Expert (if expert not provided)
            api_key: API key for the expert model
            api_base: Optional API base URL
        """
        if expert is None:
            self.expert = Expert(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
            )
        else:
            self.expert = expert

    def generate(
        self,
        task: str,
        count: int = 10,
        instruction: Optional[str] = None,
    ) -> List[Sample]:
        """Generate training samples from a task description.

        Args:
            task: Task description or domain
            count: Number of samples to generate
            instruction: Optional custom instruction prefix

        Returns:
            List of generated Sample objects
        """
        if instruction:
            full_task = f"{instruction}\n\n{task}"
        else:
            full_task = task

        samples = self.expert.produce(task=full_task, count=count)
        return samples

    def distill_from_inputs(
        self,
        inputs: List[str],
        prompt_template: str = "As an expert, generate a high-quality response to the following:\n\n{input}",
        parse_output: bool = True,
    ) -> List[Sample]:
        """Generate training samples from existing inputs.

        For each input, uses the expert to generate a high-quality output,
        creating instruction-output pairs for training.

        Args:
            inputs: List of input prompts
            prompt_template: Template for generating outputs from inputs
            parse_output: Whether to parse the output (vs treating as raw text)

        Returns:
            List of Sample objects with input_data=original input, output_data=expert response
        """
        samples = []

        for input_text in inputs:
            prompt = prompt_template.format(input=input_text)
            response = self.expert._call_litellm(prompt=prompt)

            if parse_output:
                sample = self._parse_sample_response(response)
                if sample:
                    sample.input_data = input_text
                    samples.append(sample)
            else:
                sample = Sample(
                    input_data=input_text,
                    output_data=response,
                    metadata={"source": "expert_distillation", "model": self.expert.model_name},
                )
                samples.append(sample)

        return samples

    def distill_from_texts(
        self,
        texts: List[str],
        task_description: str = "Continue the following text:",
    ) -> List[Sample]:
        """Generate continuation training samples from raw texts.

        Uses the expert to generate continuations, useful for CPT-style
        continued pre-training from existing documents.

        Args:
            texts: List of text documents
            task_description: Description of the continuation task

        Returns:
            List of Sample objects with input_data=text, output_data=continuation
        """
        samples = []

        for text in texts:
            prompt = f"{task_description}\n\n{text}"
            continuation = self.expert._call_litellm(prompt=prompt)

            sample = Sample(
                input_data=text,
                output_data=continuation,
                metadata={
                    "source": "expert_distillation",
                    "type": "continuation",
                    "model": self.expert.model_name,
                },
            )
            samples.append(sample)

        return samples

    def distill_conversations(
        self,
        topics: List[str],
        num_turns: int = 3,
    ) -> List[List[Dict[str, str]]]:
        """Generate multi-turn conversations from topics.

        Args:
            topics: List of conversation topics
            num_turns: Number of conversation turns

        Returns:
            List of conversations (each is a list of message dicts)
        """
        conversations = []

        for topic in topics:
            messages = []

            for turn in range(num_turns):
                if turn == 0:
                    prompt = f"Start a conversation about: {topic}\nProvide a user message."
                else:
                    context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    prompt = f"Continue this conversation. Provide the next {messages[-1]['role']} response.\n\nContext:\n{context}"

                response = self.expert._call_litellm(prompt=prompt)

                role = "assistant" if turn % 2 == 0 else "user"
                messages.append({"role": role, "content": response})

            conversations.append(messages)

        return conversations

    def _parse_sample_response(self, response: str) -> Optional[Sample]:
        """Parse expert response to extract input-output pair."""
        return self.expert._parse_sample_response(response)


def distill_from_expert(
    expert: Expert,
    task: str,
    count: int = 10,
) -> List[Sample]:
    """Convenience function to generate samples from an Expert.

    Args:
        expert: Expert instance to use for generation
        task: Task description
        count: Number of samples to generate

    Returns:
        List of generated Sample objects
    """
    distiller = ExpertDistiller(expert=expert)
    return distiller.generate(task=task, count=count)


def distill_from_inputs(
    inputs: List[str],
    model_name: str = "deepseek/deepseek-v3.2",
    prompt_template: str = "As an expert, generate a high-quality response to the following:\n\n{input}",
    api_key: Optional[str] = None,
) -> List[Sample]:
    """Convenience function to generate outputs for existing inputs.

    Args:
        inputs: List of input prompts
        model_name: Expert model name
        prompt_template: Template for generating outputs
        api_key: Optional API key

    Returns:
        List of Sample objects
    """
    distiller = ExpertDistiller(model_name=model_name, api_key=api_key)
    return distiller.distill_from_inputs(inputs, prompt_template)


__all__ = ["ExpertDistiller", "distill_from_expert", "distill_from_inputs"]
