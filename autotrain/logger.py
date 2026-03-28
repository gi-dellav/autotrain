"""Logging module for Autotrain with TensorBoard and WandB integration."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    log_dir: str = "./logs"
    log_level: str = "INFO"
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "autotrain"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    log_file: bool = True
    console_output: bool = True


class AutoTrainLogger:
    """
    Logger for AutoTrain with TensorBoard and WandB integration.

    Provides structured logging for training metrics, benchmark results,
    and system information.
    """

    def __init__(
        self,
        name: str = "autotrain",
        config: Optional[LoggingConfig] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            config: Logging configuration
            run_name: Optional run name (defaults to timestamp)
        """
        self.name = name
        self.config = config or LoggingConfig()
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup Python logging
        self._setup_python_logger()

        # TensorBoard writer
        self._summary_writer = None

        # WandB state
        self._wandb_initialized = False

        # Metrics storage
        self._metrics_history: Dict[str, List[float]] = {}

        # Initialize backends
        self._init_backends()

    def _setup_python_logger(self):
        """Setup Python logging handler."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.config.log_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{self.run_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Log file created: {log_file}")

    def _init_backends(self):
        """Initialize TensorBoard and WandB backends."""
        # Initialize TensorBoard
        if self.config.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = Path(self.config.log_dir) / "tensorboard" / self.run_name
                log_dir.mkdir(parents=True, exist_ok=True)

                self._summary_writer = SummaryWriter(str(log_dir))
                self.logger.info(f"TensorBoard logs: {log_dir}")
            except ImportError:
                self.logger.warning(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )
                self.config.enable_tensorboard = False

        # Initialize WandB
        if self.config.enable_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.run_name,
                tags=self.config.wandb_tags,
                config={
                    "log_dir": self.config.log_dir,
                    "log_level": self.config.log_level,
                },
            )

            self._wandb_initialized = True
            self.logger.info(f"WandB initialized: {wandb.run.url}")
        except ImportError:
            self.logger.warning("WandB not available. Install with: pip install wandb")
            self.config.enable_wandb = False
        except Exception as e:
            self.logger.warning(f"WandB initialization failed: {e}")
            self.config.enable_wandb = False

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """
        Log a scalar value.

        Args:
            name: Metric name
            value: Metric value
            step: Step/iteration number
        """
        # Store in history
        if name not in self._metrics_history:
            self._metrics_history[name] = []
        self._metrics_history[name].append(value)

        # Log to TensorBoard
        if self._summary_writer:
            self._summary_writer.add_scalar(name, value, step)

        # Log to WandB
        if self._wandb_initialized:
            import wandb

            wandb.log({name: value, "step": step})

        # Log to Python logger
        self.logger.info(f"{name}: {value:.6f} (step {step})")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values
            step: Step/iteration number
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        """
        Log text data.

        Args:
            name: Text name
            text: Text content
            step: Step/iteration number
        """
        if self._summary_writer:
            self._summary_writer.add_text(name, text, step)

        if self._wandb_initialized:
            import wandb

            wandb.log({name: wandb.Html(f"<pre>{text}</pre>"), "step": step})

        self.logger.debug(f"{name}: {text[:200]}...")

    def log_histogram(self, name: str, values: List[float], step: int) -> None:
        """
        Log histogram of values.

        Args:
            name: Histogram name
            values: List of values
            step: Step/iteration number
        """
        if self._summary_writer:
            self._summary_writer.add_histogram(name, values, step)

        if self._wandb_initialized:
            import wandb

            wandb.log({name: wandb.Histogram(values), "step": step})

    def log_benchmark_results(
        self,
        benchmark_name: str,
        accuracy: float,
        average_score: float,
        total_samples: int,
        iteration: int,
    ) -> None:
        """
        Log benchmark evaluation results.

        Args:
            benchmark_name: Name of the benchmark
            accuracy: Accuracy score
            average_score: Average score across samples
            total_samples: Total number of samples
            iteration: Current iteration
        """
        metrics = {
            f"benchmark/{benchmark_name}/accuracy": accuracy,
            f"benchmark/{benchmark_name}/average_score": average_score,
            f"benchmark/{benchmark_name}/total_samples": total_samples,
        }
        self.log_metrics(metrics, step=iteration)

    def log_training_progress(
        self,
        iteration: int,
        loss: Optional[float] = None,
        samples_count: int = 0,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Log training progress.

        Args:
            iteration: Current iteration
            loss: Current loss value
            samples_count: Total training samples
            learning_rate: Current learning rate
        """
        metrics: Dict[str, float] = {"training/samples_count": float(samples_count)}

        if loss is not None:
            metrics["training/loss"] = float(loss)

        if learning_rate is not None:
            metrics["training/learning_rate"] = float(learning_rate)

        self.log_metrics(metrics, step=iteration)

    def log_system_info(self, info: Dict[str, Any]) -> None:
        """
        Log system information.

        Args:
            info: Dictionary of system info (GPU, memory, etc.)
        """
        if self._summary_writer:
            for key, value in info.items():
                self._summary_writer.add_text(f"system/{key}", str(value), 0)

        if self._wandb_initialized:
            import wandb

            wandb.config.update(info)

        self.logger.info(f"System info: {info}")

    def log_model_config(self, config: Dict[str, Any]) -> None:
        """
        Log model configuration.

        Args:
            config: Model configuration dictionary
        """
        if self._summary_writer:
            self._summary_writer.add_text("model_config", str(config), 0)

        if self._wandb_initialized:
            import wandb

            wandb.config.update(config)

        self.logger.info(f"Model config: {config}")

    def log_checkpoint_saved(
        self,
        checkpoint_id: str,
        iteration: int,
        benchmark_accuracy: Optional[float] = None,
    ) -> None:
        """
        Log checkpoint save event.

        Args:
            checkpoint_id: Checkpoint identifier
            iteration: Iteration number
            benchmark_accuracy: Optional benchmark accuracy
        """
        self.logger.info(
            f"Checkpoint saved: {checkpoint_id} (iteration {iteration}, "
            f"accuracy: {benchmark_accuracy})"
        )

    def log_early_stopping(self, iteration: int, best_accuracy: float) -> None:
        """
        Log early stopping event.

        Args:
            iteration: Current iteration
            best_accuracy: Best achieved accuracy
        """
        self.logger.info(
            f"Early stopping at iteration {iteration} (best accuracy: {best_accuracy:.4f})"
        )

        if self._summary_writer:
            self._summary_writer.add_text(
                "early_stopping",
                f"Stopped at iteration {iteration} with best accuracy {best_accuracy:.4f}",
                iteration,
            )

    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log an error.

        Args:
            error: Exception that occurred
            context: Additional context about the error
        """
        self.logger.error(f"{context}: {error}", exc_info=True)

        if self._wandb_initialized:
            import wandb

            wandb.alert(
                title="AutoTrain Error",
                text=f"{context}: {error}",
                level=2,  # ERROR level
            )

    def log_completion(self, summary: Dict[str, Any]) -> None:
        """
        Log training completion summary.

        Args:
            summary: Training summary dictionary
        """
        self.logger.info("=== Training Complete ===")
        self.logger.info(f"Iterations completed: {summary.get('iterations_completed', 0)}")
        self.logger.info(f"Total samples: {summary.get('total_samples', 0)}")
        self.logger.info(f"Best accuracy: {summary.get('best_accuracy', 0):.4f}")

        if self._summary_writer:
            self._summary_writer.add_text(
                "training_summary",
                str(summary),
                summary.get("iterations_completed", 0),
            )

        if self._wandb_initialized:
            import wandb

            wandb.summary.update(summary)

    def close(self) -> None:
        """Close all logging backends."""
        if self._summary_writer:
            self._summary_writer.close()
            self.logger.info("TensorBoard writer closed")

        if self._wandb_initialized:
            import wandb

            wandb.finish()
            self.logger.info("WandB run finished")

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        Get the history of logged metrics.

        Returns:
            Dictionary mapping metric names to list of values
        """
        return self._metrics_history.copy()

    def get_metric_average(self, name: str, last_n: Optional[int] = None) -> float:
        """
        Get average of a metric over recent values.

        Args:
            name: Metric name
            last_n: Number of recent values to average (None for all)

        Returns:
            Average value
        """
        if name not in self._metrics_history:
            return 0.0

        values = self._metrics_history[name]
        if last_n:
            values = values[-last_n:]

        return sum(values) / len(values) if values else 0.0


def get_logger(
    name: str = "autotrain",
    log_dir: str = "./logs",
    enable_tensorboard: bool = True,
    enable_wandb: bool = False,
    wandb_project: str = "autotrain",
    run_name: Optional[str] = None,
) -> AutoTrainLogger:
    """
    Convenience function to create a logger.

    Args:
        name: Logger name
        log_dir: Directory for log files
        enable_tensorboard: Enable TensorBoard logging
        enable_wandb: Enable WandB logging
        wandb_project: WandB project name
        run_name: Optional run name

    Returns:
        Configured AutoTrainLogger instance
    """
    config = LoggingConfig(
        log_dir=log_dir,
        enable_tensorboard=enable_tensorboard,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
    )

    return AutoTrainLogger(name=name, config=config, run_name=run_name)
