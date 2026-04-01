"""Model training and expert management."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from autotrain.components import ExpertWeight
from autotrain.data_types import Sample

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.core.model import Model
    from autotrain.core.vision_model import VisionModel
    from autotrain.expert import Expert, VisionExpert
    from autotrain.templates import InstructionTemplate


def _init_components(
    model: "Model",
    experts: Optional[list[tuple["Expert", float]]] = None,
):
    """Initialize pipeline components."""
    from autotrain.components import Checker, Producer, Solver, Splitter

    model._producer = Producer(
        model=model,
        prompt=model.prompts.producer,
        inference_config=model.inference_config,
    )

    # Initialize solver with multiple experts
    expert_weights = []
    if experts:
        for expert in experts:
            expert_weights.append(ExpertWeight(expert=expert, weight=expert.production_rate))

    model._solver = Solver(
        model=model,
        prompt=model.prompts.solver,
        inference_config=model.inference_config,
        experts=expert_weights if expert_weights else None,  # type: ignore[arg-type]
    )

    # Get experts list for other components
    expert_list = [ew.expert for ew in expert_weights] if expert_weights else []

    model._splitter = Splitter(
        model=model,
        prompt=model.prompts.splitter,
        inference_config=model.inference_config,
        experts=expert_list,
    )

    if model.enable_checker:
        model._checker = Checker(
            model=model,
            prompt=model.prompts.checker,
            inference_config=model.inference_config,
            experts=expert_list,
        )


def _fine_tune(model: "Model", resume_from_checkpoint: bool = False) -> None:
    """Fine-tune the model on accumulated training data from scratch."""
    if not model._training_data:
        return

    try:
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel

        # Prepare dataset
        train_dataset = Dataset.from_list(model._training_data)

        from autotrain.templates import format_sample

        # Format dataset for instruction tuning
        def format_example(example):
            return {
                "text": format_sample(
                    instruction=example.get("input", ""),
                    output=example.get("output", ""),
                    template=model._template,
                    model_name=model.model_name,
                )
            }

        train_dataset = train_dataset.map(format_example)

        # Setup PEFT from scratch
        use_gc = model._scalable_config.gradient_checkpointing
        model._fast_model = FastLanguageModel.get_peft_model(
            model=model._fast_model,
            r=model._peft_config.r,
            target_modules=model._peft_config.get_target_modules(),
            lora_alpha=model._peft_config.lora_alpha,
            lora_dropout=model._peft_config.lora_dropout,
            bias=model._peft_config.bias,
            use_gradient_checkpointing="unsloth" if use_gc else False,
        )

        # Auto-tune batch size if enabled
        batch_size = model._training_config.batch_size
        if model._scalable_config.batch_size_auto_tune:
            batch_size = model.auto_tune_batch_size()

        # Determine mixed precision
        fp16 = model._scalable_config.mixed_precision == "fp16"
        bf16 = model._scalable_config.mixed_precision == "bf16"

        # Training arguments
        training_args = TrainingArguments(  # type: ignore[call-arg]
            output_dir="./training_output",
            num_train_epochs=model._training_config.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=model._training_config.gradient_accumulation_steps,
            learning_rate=model._training_config.learning_rate,
            weight_decay=model._training_config.weight_decay,
            warmup_ratio=model._training_config.warmup_ratio,
            max_grad_norm=model._training_config.max_grad_norm,
            logging_steps=model._training_config.logging_steps,
            save_strategy=model._training_config.save_strategy,
            save_steps=model._training_config.save_steps,
            eval_strategy=model._training_config.eval_strategy,
            eval_steps=model._training_config.eval_steps,
            save_total_limit=model._training_config.save_total_limit,
            seed=model._training_config.seed,
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=model._scalable_config.num_workers,
            pin_memory=model._scalable_config.pin_memory,
            report_to="none",
            lr_scheduler_type=model._training_config.scheduler_type,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model._fast_model,  # type: ignore[arg-type]
            tokenizer=model._tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            args=training_args,
        )  # type: ignore[call-arg]

        # Train with optional checkpoint resume
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except ImportError as e:
        print(f"Fine-tuning error (missing dependency): {e}")
    except Exception as e:
        print(f"Fine-tuning error: {e}")


def train(
    model: "Model",
    k: Optional[int] = 100,
    i: int = 10,
    experts: Optional[list["Expert"]] = None,
    initial_samples: Optional[list[Sample]] = None,
    benchmark: Optional["Benchmark"] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.01,
    checkpoint_every: int = 1,
    keep_best_model: bool = True,
    resume_from_checkpoint: bool = False,
    template: Optional[Union["InstructionTemplate", str]] = None,
    tools: Optional[list] = None,
    enable_tools: bool = False,
    max_tool_calls: int = 10,
    tool_choice: Optional[str] = None,
) -> dict:
    """
    Train the model through iterations.

    Args:
        k: Number of samples to use for fine-tuning
        i: Number of iterations
        experts: List of Expert instances for multi-expert setup (uses each expert's production_rate)
        initial_samples: Optional initial dataset to start with
        benchmark: Optional benchmark for evaluation
        early_stopping: Enable early stopping based on benchmark
        early_stopping_patience: Iterations without improvement before stopping
        early_stopping_threshold: Minimum improvement to count
        checkpoint_every: Save checkpoint every N iterations
        keep_best_model: Keep the best model even if not last iteration
        resume_from_checkpoint: Resume from latest checkpoint if available
        template: Optional template for instruction formatting
        tools: List of tools to register for tool calling (e.g., [python, calculator])
        enable_tools: Enable tool calling during training (default: False)
        max_tool_calls: Maximum number of tool calls per generation (default: 10)
        tool_choice: Tool choice option ("auto", "none", or specific tool name)

    Returns:
        Training summary dictionary

    Example:
        from autotrain.tools import python, calculator

        model = Model(model_name="unsloth/Qwen3.5-27B-GGUF")
        model.load_model()

        # Train with tool calling enabled
        model.train(
            k=100,
            i=10,
            tools=[python, calculator],
            enable_tools=True,
            max_tool_calls=5,
        )
    """
    # Validate parameters
    if k is not None and k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if i <= 0:
        raise ValueError(f"i (iterations) must be positive, got {i}")
    if early_stopping_patience <= 0:
        raise ValueError(f"early_stopping_patience must be positive, got {early_stopping_patience}")
    if early_stopping_threshold < 0:
        raise ValueError(
            f"early_stopping_threshold must be non-negative, got {early_stopping_threshold}"
        )
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be non-negative, got {checkpoint_every}")

    # Validate experts
    if experts:
        for idx, expert in enumerate(experts):
            if not hasattr(expert, "model_name"):
                raise ValueError(f"Expert at index {idx} must have a 'model_name' attribute")

    # Set template if provided
    if template is not None:
        model.set_template(template)

    # Setup tools for tool calling
    if tools or enable_tools:
        if tools:
            for tool in tools:
                model.add_tool(tool)
            print(f"Registered {len(tools)} tools: {model.list_tools()}")

        if enable_tools:
            print(
                f"Tool calling enabled (max_tool_calls={max_tool_calls}, tool_choice={tool_choice})"
            )

    # Handle resume from checkpoint
    if resume_from_checkpoint:
        checkpoints = model._checkpoint_manager.list_checkpoints()
        if checkpoints:
            print(f"Resuming from latest checkpoint: {checkpoints[-1].checkpoint_id}")
            model.load_checkpoint(checkpoint_id=checkpoints[-1].checkpoint_id)
            start_iteration = model._current_iteration + 1
        else:
            start_iteration = 0
    else:
        start_iteration = 0

    if k is None:
        k = len(initial_samples) if initial_samples else 10

    if initial_samples:
        model._samples.extend(initial_samples)

    if not model._is_model_loaded:
        try:
            print("Loading model...")
            model.load_model()
        except ImportError as e:
            raise ImportError(
                f"Failed to load model: {e}\nMake sure unsloth is installed: pip install unsloth"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model.model_name}: {e}")

    # Set benchmark if provided
    if benchmark:
        model._benchmark = benchmark

    # Validate benchmark configuration
    if early_stopping and model._benchmark and model._benchmark.sample_count == 0:
        print(
            "Warning: Early stopping enabled but benchmark has no samples. "
            "Disabling early stopping."
        )
        early_stopping = False

    if model._benchmark and model._benchmark.sample_count > 0:
        print(f"Benchmark configured with {model._benchmark.sample_count} samples")

    _init_components(model, experts=experts)

    best_accuracy = 0.0

    training_summary: Dict[str, Any] = {
        "iterations_completed": 0,
        "total_samples": 0,
        "benchmark_history": [],  # type: ignore[var-annotated]
        "stopped_early": False,
        "best_iteration": -1,
        "best_accuracy": 0.0,
    }

    for iteration in range(start_iteration, start_iteration + i):
        model._current_iteration = iteration
        print(f"\n=== Iteration {iteration + 1}/{start_iteration + i} ===")

        # Reload the base model from scratch for higher-quality fine-tuning
        if iteration > start_iteration:
            print("Reloading base model from scratch for this iteration...")
            model._unload_model()
            model._is_model_loaded = False
            model.load_model()
            # Re-initialize components with the reloaded model
            _init_components(model, experts=experts)

        # Producer: Generate input samples
        target_samples = k * model.sample_multiplier
        assert model._producer is not None, "Producer should be initialized"
        input_samples = model._producer.generate(target_samples)
        print(f"Produced {len(input_samples)} input samples")

        # Solver: LLM solves the input samples
        assert model._solver is not None, "Solver should be initialized"
        output_samples = model._solver.solve(input_samples)
        print(f"Solved {len(output_samples)} samples")

        # Checker: Optional verification
        if model.enable_checker and model._checker:
            verified_samples = model._checker.verify(output_samples)
            rejected_count = len(output_samples) - len(verified_samples)
            rejection_rate = (rejected_count / len(output_samples) * 100) if output_samples else 0
            print(
                f"Verified {len(verified_samples)} samples ({rejected_count} rejected, "
                f"{rejection_rate:.1f}% rejection rate)"
            )
            output_samples = verified_samples

        # Splitter: Select useful samples based on sample_multiplier
        assert model._splitter is not None, "Splitter should be initialized"
        selected_samples = model._splitter.select(output_samples, target_count=k)
        print(f"Selected {len(selected_samples)} samples")

        # Add to training data
        model._samples.extend(selected_samples)
        model._training_data.extend(
            [{"input": s.input_data, "output": s.output_data} for s in selected_samples]
        )

        # Fine-tune the model (with optional step-level checkpoint resume)
        _fine_tune(model, resume_from_checkpoint=resume_from_checkpoint)
        print(f"Fine-tuning completed for iteration {iteration + 1}")

        # Evaluate on benchmark
        current_accuracy = 0.0
        if model._benchmark:
            metrics = model._benchmark.evaluate(
                model=model, iteration=iteration, inference_config=model.inference_config
            )
            current_accuracy = metrics.accuracy
            print(f"Benchmark accuracy: {current_accuracy:.4f}")

            training_summary["benchmark_history"].append(
                {
                    "iteration": iteration,
                    "accuracy": current_accuracy,
                    "average_score": metrics.average_score,
                }
            )

            # Track best
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                training_summary["best_iteration"] = iteration
                training_summary["best_accuracy"] = current_accuracy

                # Save best model
                if keep_best_model:
                    model.save_checkpoint(
                        iteration=iteration,
                        metadata={"is_best": True, "accuracy": current_accuracy},
                    )

                # Early stopping check
                if early_stopping and model._benchmark:
                    if model._benchmark.has_stagnated(
                        threshold=early_stopping_threshold, iterations=early_stopping_patience
                    ):
                        print(
                            f"\nEarly stopping triggered: no significant improvement in "
                            f"{early_stopping_patience} iterations"
                        )
                        training_summary["stopped_early"] = True
                        break

        # Save checkpoint
        if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
            model.save_checkpoint(iteration=iteration, metadata={"accuracy": current_accuracy})

        training_summary["iterations_completed"] = iteration + 1
        training_summary["total_samples"] = len(model._samples)

    # Final summary
    print("\n=== Training Summary ===")
    print(f"Iterations completed: {training_summary['iterations_completed']}")
    print(f"Total samples: {training_summary['total_samples']}")
    print(f"Best iteration: {training_summary['best_iteration']}")
    print(f"Best accuracy: {training_summary['best_accuracy']:.4f}")

    if training_summary["stopped_early"]:
        print("Training stopped early due to stagnation")

    # Export benchmark results
    if model._benchmark:
        from pathlib import Path

        results_path = Path(model._checkpoint_manager.checkpoint_dir) / "benchmark_results.json"
        model._benchmark.export_results(str(results_path))

    return training_summary


def train_vision_model(
    model: "VisionModel",
    k: int = 10,
    i: int = 5,
    experts: Optional[list["VisionExpert"]] = None,
    initial_samples: Optional[list] = None,
    benchmark: Optional["Benchmark"] = None,
    early_stopping: bool = True,
    checkpoint_every: int = 1,
    resume_from_checkpoint: bool = False,
) -> dict:
    """
    Train a VisionModel using the self-tuning loop.

    Args:
        model: The VisionModel to train
        k: Number of samples per iteration
        i: Number of iterations
        experts: Optional list of VisionExpert instances (uses each expert's production_rate)
        initial_samples: Optional initial vision samples
        benchmark: Optional evaluation benchmark
        early_stopping: Stop if no improvement
        checkpoint_every: Save checkpoint every N iterations
        resume_from_checkpoint: Resume from latest checkpoint

    Returns:
        Training summary dictionary
    """
    from autotrain.components import ExpertWeight
    from autotrain.components.vision_producer import VisionProducer
    from autotrain.data_types import VisionSample
    from autotrain.expert import VisionExpert

    if k is not None and k <= 0:
        raise ValueError("k must be positive")

    if i is not None and i <= 0:
        raise ValueError("i must be positive")

    if k is None:
        k = len(initial_samples) if initial_samples else 10
    if i is None:
        i = 5

    print(f"Starting vision model training: k={k}, iterations={i}")

    if initial_samples:
        for sample in initial_samples:
            if isinstance(sample, VisionSample):
                model.add_sample(sample)

    expert_weights = []
    if experts:
        for expert in experts:
            if isinstance(expert, VisionExpert):
                expert_weights.append(ExpertWeight(expert=expert, weight=expert.production_rate))

    vision_producer = VisionProducer(
        model=model,
        prompt=model.prompts.producer,
        inference_config=model.inference_config,
    )

    best_accuracy = 0.0
    stopped_early = False
    stagnation_count = 0
    max_stagnation = 3

    sample_multiplier = model.sample_multiplier

    for iteration in range(i):
        model._current_iteration = iteration + 1
        print(f"\n=== Iteration {iteration + 1}/{i} ===")

        if resume_from_checkpoint and model._checkpoint_manager.has_checkpoint():
            print(f"Resuming from checkpoint...")
            continue

        samples_to_generate = k * sample_multiplier
        input_samples = vision_producer.generate(count=samples_to_generate)
        print(f"Generated {len(input_samples)} input samples")

        for sample in input_samples:
            model.add_sample(sample)

        _fine_tune_vision(model)

        if benchmark:
            accuracy = benchmark.evaluate(model)
            print(f"Benchmark accuracy: {accuracy:.4f}")

            model._benchmark_history.append(
                {
                    "iteration": iteration + 1,
                    "accuracy": accuracy,
                }
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                stagnation_count = 0
                model._checkpoint_manager.save_checkpoint(model, accuracy, iteration + 1)
                print(f"New best accuracy! Saved checkpoint.")
            else:
                stagnation_count += 1
                print(f"No improvement for {stagnation_count} iteration(s)")

                if early_stopping and stagnation_count >= max_stagnation:
                    print("Early stopping triggered")
                    stopped_early = True
                    break

        elif checkpoint_every and (iteration + 1) % checkpoint_every == 0:
            model._checkpoint_manager.save_checkpoint(model, 0.0, iteration + 1)
            print(f"Checkpoint saved at iteration {iteration + 1}")

    training_summary = {
        "iterations_completed": i if not stopped_early else iteration + 1,
        "total_samples": len(model._training_data),
        "best_accuracy": best_accuracy,
        "stopped_early": stopped_early,
    }

    print(f"\nTraining complete!")
    print(f"Iterations: {training_summary['iterations_completed']}")
    print(f"Total samples used: {training_summary['total_samples']}")

    if benchmark:
        print(f"Best accuracy: {training_summary['best_accuracy']:.4f}")

    if training_summary["stopped_early"]:
        print("Training stopped early due to stagnation")

    return training_summary


def _fine_tune_vision(model: "VisionModel", resume_from_checkpoint: bool = False) -> None:
    """Fine-tune the vision model on accumulated training data."""
    if not model._training_data:
        return

    try:
        from datasets import Dataset
        from transformers import TrainingArguments

        from unsloth import FastVisionModel

        train_dataset = Dataset.from_list(model._training_data)

        def format_example(example):
            if hasattr(model, "_template") and model._template:
                messages = example.get("messages", [])
                if not messages and "input_data" in example:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": example["input_data"]}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": example["output_data"]}],
                        },
                    ]
                text = model._template.format_training_sample(
                    instruction=example.get("input_data", ""),
                    output=example.get("output_data", ""),
                )
            else:
                text = f"### Instruction:\n{example.get('input_data', '')}\n\n### Response:\n{example.get('output_data', '')}"
            return {"text": text}

        train_dataset = train_dataset.map(format_example)

        use_gc = model._scalable_config.gradient_checkpointing
        model._fast_model = FastVisionModel.get_peft_model(
            model=model._fast_model,
            r=model._peft_config.r,
            target_modules=model._peft_config.get_target_modules(),
            lora_alpha=model._peft_config.lora_alpha,
            lora_dropout=model._peft_config.lora_dropout,
            bias=model._peft_config.bias,
            use_gradient_checkpointing=use_gc if use_gc else "unsloth",
        )

        output_dir = model._checkpoint_manager.checkpoint_dir
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=model._training_config.batch_size,
            gradient_accumulation_steps=model._training_config.gradient_accumulation_steps,
            learning_rate=model._training_config.learning_rate,
            num_train_epochs=model._training_config.epochs,
            max_steps=model._training_config.max_steps,
            warmup_ratio=model._training_config.warmup_ratio,
            logging_steps=model._training_config.logging_steps,
            save_steps=model._training_config.save_steps,
            save_total_limit=model._training_config.save_total_limit,
            fp16=model._scalable_config.mixed_precision == "fp16",
            bf16=model._scalable_config.mixed_precision == "bf16",
            gradient_checkpointing=use_gc,
            report_to=[],
            seed=42,
            lr_scheduler_type=model._training_config.scheduler_type,
        )

        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model._fast_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=model._tokenizer,
            dataset_text_field="text",
        )

        print(f"Fine-tuning with {len(train_dataset)} samples...")
        trainer.train()

        model._is_trained = True
        print("Fine-tuning complete")

    except ImportError as e:
        print(f"Warning: Missing dependency for training: {e}")
        print("Training data accumulated but fine-tuning skipped")


__all__ = ["train", "_init_components", "_fine_tune", "train_vision_model", "_fine_tune_vision"]
