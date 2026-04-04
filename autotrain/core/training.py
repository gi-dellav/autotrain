"""Model training and expert management."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from pathlib import Path

from autotrain.components import ExpertWeight
from autotrain.data_types import Sample, VisionSample

if TYPE_CHECKING:
    from autotrain.benchmark import Benchmark
    from autotrain.core.base_model import BaseModel
    from autotrain.expert import Expert


def _init_components(
    model: "BaseModel",
    experts: Optional[list] = None,
):
    """Initialize pipeline components."""
    from autotrain.components import Checker, Producer, Solver, Splitter
    from autotrain.components.vision_producer import VisionProducer

    # Check if it's a vision model
    is_vision = hasattr(model, "processor")

    if is_vision:
        model._producer = VisionProducer(
            model=model,  # type: ignore
            prompt=model.prompts.producer,  # type: ignore
            inference_config=model.inference_config,
        )
    else:
        model._producer = Producer(
            model=model,  # type: ignore
            prompt=model.prompts.producer,  # type: ignore
            inference_config=model.inference_config,
        )

    # Initialize solver with multiple experts
    expert_weights = []
    if experts:
        for exp_entry in experts:
            if isinstance(exp_entry, tuple):
                exp, weight = exp_entry
                expert_weights.append(ExpertWeight(expert=exp, weight=weight))
            elif hasattr(exp_entry, "production_rate"):
                expert_weights.append(
                    ExpertWeight(expert=exp_entry, weight=exp_entry.production_rate)
                )
            else:
                expert_weights.append(ExpertWeight(expert=exp_entry, weight=1.0))

    model._solver = Solver(
        model=model,  # type: ignore
        prompt=model.prompts.solver,  # type: ignore
        inference_config=model.inference_config,
        experts=expert_weights if expert_weights else None,  # type: ignore
    )

    expert_list = [ew.expert for ew in expert_weights] if expert_weights else []

    model._splitter = Splitter(
        model=model,  # type: ignore
        prompt=model.prompts.splitter,  # type: ignore
        inference_config=model.inference_config,
        experts=expert_list,
    )

    if model.enable_checker:
        model._checker = Checker(
            model=model,  # type: ignore
            prompt=model.prompts.checker,  # type: ignore
            inference_config=model.inference_config,
            experts=expert_list,
            rewrite_mode=model.checker_rewrite_mode,
        )


def _prune_old_training_data(
    model: "BaseModel", current_iteration: int, keep_last_n_iters: int
) -> None:
    """Remove training data from iterations older than keep_last_n_iters."""
    if keep_last_n_iters <= 0:
        return

    min_iteration = current_iteration - keep_last_n_iters + 1

    # Prune _training_data, but always keep initial samples (iteration == -1)
    model._training_data = [
        sample
        for sample in model._training_data
        if sample.get("iteration", 0) == -1 or sample.get("iteration", 0) >= min_iteration
    ]

    print(
        f"Pruned training data to keep last {keep_last_n_iters} iterations (min_iteration={min_iteration})"
    )
    print(f"Remaining training samples: {len(model._training_data)}")


def _fine_tune(model: "BaseModel", iteration: int, resume_from_checkpoint: bool = False) -> None:
    """Fine-tune the model on accumulated training data."""
    from autotrain.utils.function_evaluator import (
        evaluate_batch_size,
        evaluate_epochs,
        evaluate_learning_rate,
        evaluate_lora_alpha,
        evaluate_lora_dropout,
        evaluate_lora_rank,
        evaluate_weight_decay,
    )

    iteration = max(0, iteration)
    epochs = evaluate_epochs(
        model._training_config.epochs_fn or model._training_config.epochs, iteration
    )
    learning_rate = evaluate_learning_rate(
        model._training_config.learning_rate_fn or model._training_config.learning_rate, iteration
    )
    lora_alpha = evaluate_lora_alpha(
        model._peft_config.lora_alpha_fn or model._peft_config.lora_alpha, iteration
    )
    lora_dropout = evaluate_lora_dropout(
        model._peft_config.lora_dropout_fn or model._peft_config.lora_dropout, iteration
    )
    lora_rank = evaluate_lora_rank(
        model._peft_config.lora_rank_fn or model._peft_config.r, iteration
    )
    weight_decay = evaluate_weight_decay(
        model._training_config.weight_decay_fn or model._training_config.weight_decay, iteration
    )

    if not model._training_data:
        return

    is_vision = hasattr(model, "processor")

    try:
        from datasets import Dataset  # type: ignore[import-untyped]
        from transformers import TrainingArguments  # type: ignore[import-untyped]
        from trl import SFTTrainer  # type: ignore[import-untyped]

        train_dataset = Dataset.from_list(model._training_data)

        from autotrain.templates import apply_chat_template, format_sample

        def format_example(example):
            if "messages" in example:
                return {
                    "text": apply_chat_template(
                        messages=example["messages"],
                        template=getattr(model, "_template", None),
                    )
                }
            return {
                "text": format_sample(
                    instruction=example.get("input", example.get("input_data", "")),
                    output=example.get("output", example.get("output_data", "")),
                    template=getattr(model, "_template", None),
                    model_name=model.model_name,
                )
            }

        train_dataset = train_dataset.map(format_example)

        use_gc = model._scalable_config.gradient_checkpointing

        if is_vision:
            from unsloth import FastVisionModel  # type: ignore[import-untyped]

            model._fast_model = FastVisionModel.get_peft_model(
                model=model._fast_model,
                r=lora_rank,
                target_modules=model._peft_config.get_target_modules(),
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=model._peft_config.bias,
                use_gradient_checkpointing="unsloth" if use_gc else False,
            )
        else:
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]

            model._fast_model = FastLanguageModel.get_peft_model(
                model=model._fast_model,
                r=lora_rank,
                target_modules=model._peft_config.get_target_modules(),
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=model._peft_config.bias,
                use_gradient_checkpointing="unsloth" if use_gc else False,
            )

        batch_size = evaluate_batch_size(
            model._training_config.batch_size_fn or model._training_config.batch_size, iteration
        )
        if not is_vision and model._scalable_config.batch_size_auto_tune:
            batch_size = model.auto_tune_batch_size()  # type: ignore

        fp16 = model._scalable_config.mixed_precision == "fp16"
        bf16 = model._scalable_config.mixed_precision == "bf16"

        training_args = TrainingArguments(
            output_dir="./training_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=model._training_config.gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
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
            report_to="none",
            lr_scheduler_type=model._training_config.scheduler_type,
        )

        trainer = SFTTrainer(
            model=model._fast_model,  # type: ignore
            tokenizer=model._tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",  # type: ignore
            args=training_args,
        )  # type: ignore

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except Exception as e:
        print(f"Fine-tuning error: {e}")


def _fine_tune_vision(
    model: "BaseModel", iteration: int, resume_from_checkpoint: bool = False
) -> None:
    """Compatibility alias for _fine_tune."""
    _fine_tune(model, iteration, resume_from_checkpoint)


def train(
    model: "BaseModel",
    k: Optional[int] = 100,
    i: int = 10,
    experts: Optional[list] = None,
    initial_samples: Optional[list] = None,
    benchmark: Optional["Benchmark"] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.01,
    checkpoint_every: int = 1,
    keep_best_model: bool = True,
    resume_from_checkpoint: bool = False,
    template: Optional[Any] = None,
    tools: Optional[list] = None,
    enable_tools: bool = False,
    max_tool_calls: int = 10,
    tool_choice: Optional[str] = None,
) -> dict:
    """Unified training loop for both standard and vision models."""
    if i <= 0:
        raise ValueError("i must be positive")

    if template is not None and hasattr(model, "set_template"):
        model.set_template(template)  # type: ignore

    if tools or enable_tools:
        if tools:
            for tool in tools:
                model.add_tool(tool)
        print(f"Tool calling enabled: {model.list_tools()}")

    if resume_from_checkpoint:
        checkpoints = model.list_checkpoints()
        if checkpoints:
            model.load_checkpoint(checkpoint_id=checkpoints[-1].checkpoint_id)
            start_iteration = model._current_iteration + 1
        else:
            start_iteration = 0
    else:
        start_iteration = 0

    if initial_samples:
        for s in initial_samples:
            model.add_sample(s)  # type: ignore
            # Mark initial samples as iteration -1 (always kept)
            if model._training_data:
                model._training_data[-1]["iteration"] = -1

    if not model.is_loaded:
        model.load_model()

    if benchmark:
        model.set_benchmark(benchmark)

    _init_components(model, experts=experts or model.get_experts())

    best_accuracy = 0.0
    summary: dict[str, Any] = {
        "iterations_completed": 0,
        "total_samples": 0,
        "benchmark_history": [],
        "stopped_early": False,
        "best_iteration": -1,
        "best_accuracy": 0.0,
    }

    for iteration in range(start_iteration, start_iteration + i):
        model._current_iteration = iteration
        print(f"\n=== Iteration {iteration + 1}/{start_iteration + i} ===")

        if iteration > start_iteration:
            model._unload_model()
            model.load_model()
            _init_components(model, experts=experts or model.get_experts())

        # Pipeline steps
        target_k = k or (len(initial_samples) if initial_samples else 10)
        input_samples = model._producer.generate(target_k * model.sample_multiplier)  # type: ignore
        print(f"Produced {len(input_samples)} samples")

        output_samples = model._solver.solve(input_samples)  # type: ignore
        print(f"Solved {len(output_samples)} samples")

        if model.enable_checker and model._checker:
            output_samples = model._checker.verify(output_samples)
            print(f"Verified {len(output_samples)} samples")

        selected_samples = model._splitter.select(output_samples, target_count=target_k)  # type: ignore
        print(f"Selected {len(selected_samples)} samples")

        # Prune old training data if keep_last_n_iters is set
        if model._training_config.keep_last_n_iters is not None:
            _prune_old_training_data(model, iteration, model._training_config.keep_last_n_iters)

        # Add to training data with iteration tracking
        for s in selected_samples:
            model.add_sample(s)  # type: ignore
            # Add iteration metadata to the last added sample
            if model._training_data:
                model._training_data[-1]["iteration"] = iteration

        _fine_tune(model, iteration, resume_from_checkpoint=resume_from_checkpoint)

        # Evaluation
        curr_acc = 0.0
        if model.get_benchmark():
            metrics = model.get_benchmark().evaluate(model=model, iteration=iteration, inference_config=model.inference_config)  # type: ignore
            curr_acc = metrics.accuracy
            print(f"Accuracy: {curr_acc:.4f}")
            summary["benchmark_history"].append({"iteration": iteration, "accuracy": curr_acc})

            if curr_acc > best_accuracy:
                best_accuracy = curr_acc
                summary["best_iteration"] = iteration
                summary["best_accuracy"] = curr_acc
                if keep_best_model:
                    model.save_checkpoint(
                        iteration, metadata={"is_best": True, "accuracy": curr_acc}
                    )

                if early_stopping and model.get_benchmark().has_stagnated(threshold=early_stopping_threshold, iterations=early_stopping_patience):  # type: ignore
                    summary["stopped_early"] = True
                    break

        if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
            model.save_checkpoint(iteration, metadata={"accuracy": curr_acc})

        summary["iterations_completed"] = iteration + 1
        summary["total_samples"] = len(model.get_samples())

    # Export benchmark results
    benchmark = model.get_benchmark()
    if benchmark:
        results_path = Path(model._checkpoint_manager.checkpoint_dir) / "benchmark_results.json"
        benchmark.export_results(str(results_path))

    return summary


def train_vision_model(model: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility wrapper for vision training."""
    return train(model, **kwargs)


__all__ = ["train", "_init_components", "_fine_tune", "train_vision_model", "_fine_tune_vision"]
