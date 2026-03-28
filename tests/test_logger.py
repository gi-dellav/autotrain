"""Tests for AutoTrain logger module."""

from unittest.mock import MagicMock, patch

from autotrain.logger import (
    AutoTrainLogger,
    LoggingConfig,
    get_logger,
)


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_config(self):
        """Test default logging config values."""
        config = LoggingConfig()

        assert config.log_dir == "./logs"
        assert config.log_level == "INFO"
        assert config.enable_tensorboard is True
        assert config.enable_wandb is False
        assert config.wandb_project == "autotrain"
        assert config.wandb_entity is None
        assert config.log_file is True
        assert config.console_output is True

    def test_custom_config(self):
        """Test custom logging config."""
        config = LoggingConfig(
            log_dir="./custom_logs",
            log_level="DEBUG",
            enable_tensorboard=False,
            enable_wandb=True,
            wandb_project="my_project",
            wandb_entity="my_entity",
            wandb_tags=["tag1", "tag2"],
            log_file=False,
            console_output=False,
        )

        assert config.log_dir == "./custom_logs"
        assert config.log_level == "DEBUG"
        assert config.enable_tensorboard is False
        assert config.enable_wandb is True
        assert config.wandb_project == "my_project"
        assert config.wandb_entity == "my_entity"
        assert config.wandb_tags == ["tag1", "tag2"]


class TestAutoTrainLogger:
    """Tests for AutoTrainLogger class."""

    def test_logger_init_default(self, temp_dir):
        """Test logger initialization with defaults."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        assert logger.name == "autotrain"
        assert logger.config == config
        assert logger._metrics_history == {}

    def test_logger_init_custom_name(self, temp_dir):
        """Test logger initialization with custom name."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(name="test_logger", config=config)

        assert logger.name == "test_logger"

    def test_logger_creates_log_dir(self, temp_dir):
        """Test that logger creates log directory."""
        log_dir = temp_dir / "test_logs"
        config = LoggingConfig(log_dir=str(log_dir), enable_tensorboard=False, enable_wandb=False)

        AutoTrainLogger(config=config)

        assert log_dir.exists()

    def test_logger_creates_log_file(self, temp_dir):
        """Test that logger creates log file."""
        log_dir = temp_dir / "test_logs"
        config = LoggingConfig(
            log_dir=str(log_dir),
            enable_tensorboard=False,
            enable_wandb=False,
            log_file=True,
        )

        AutoTrainLogger(config=config)

        # Log files should be created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) >= 1

    def test_log_scalar(self, temp_dir):
        """Test logging scalar values."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.log_scalar("test_metric", 0.95, step=1)

        # Check history
        assert "test_metric" in logger._metrics_history
        assert logger._metrics_history["test_metric"] == [0.95]

    def test_log_scalar_multiple(self, temp_dir):
        """Test logging multiple scalar values."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        logger.log_scalar("loss", 0.5, step=1)
        logger.log_scalar("loss", 0.4, step=2)
        logger.log_scalar("loss", 0.3, step=3)

        assert logger._metrics_history["loss"] == [0.5, 0.4, 0.3]

    def test_log_metrics(self, temp_dir):
        """Test logging multiple metrics at once."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        metrics = {
            "accuracy": 0.9,
            "loss": 0.5,
            "f1_score": 0.85,
        }

        logger.log_metrics(metrics, step=1)

        assert "accuracy" in logger._metrics_history
        assert "loss" in logger._metrics_history
        assert "f1_score" in logger._metrics_history

    def test_log_text(self, temp_dir):
        """Test logging text data."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.log_text("sample_text", "This is a test sample", step=1)

    def test_log_histogram(self, temp_dir):
        """Test logging histogram."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        values = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Should not raise
        logger.log_histogram("test_histogram", values, step=1)

    def test_log_benchmark_results(self, temp_dir):
        """Test logging benchmark results."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.log_benchmark_results(
            benchmark_name="math_benchmark",
            accuracy=0.85,
            average_score=0.82,
            total_samples=100,
            iteration=5,
        )

        # Check metrics were recorded
        assert "benchmark/math_benchmark/accuracy" in logger._metrics_history
        assert "benchmark/math_benchmark/average_score" in logger._metrics_history

    def test_log_training_progress(self, temp_dir):
        """Test logging training progress."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.log_training_progress(
            iteration=10,
            loss=0.45,
            samples_count=500,
            learning_rate=1e-4,
        )

        assert "training/samples_count" in logger._metrics_history
        assert "training/loss" in logger._metrics_history
        assert "training/learning_rate" in logger._metrics_history

    def test_log_system_info(self, temp_dir):
        """Test logging system information."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        info = {
            "gpu": "NVIDIA A100",
            "memory": "40GB",
            "cuda_version": "11.8",
        }

        # Should not raise
        logger.log_system_info(info)

    def test_log_model_config(self, temp_dir):
        """Test logging model configuration."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        config_dict = {
            "model_name": "llama-3-8b",
            "learning_rate": 1e-4,
            "batch_size": 4,
        }

        # Should not raise
        logger.log_model_config(config_dict)

    def test_log_checkpoint_saved(self, temp_dir):
        """Test logging checkpoint save event."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.log_checkpoint_saved(
            checkpoint_id="ckpt_1",
            iteration=5,
            benchmark_accuracy=0.85,
        )

    def test_log_early_stopping(self, temp_dir):
        """Test logging early stopping event."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.log_early_stopping(iteration=10, best_accuracy=0.92)

    def test_log_error(self, temp_dir):
        """Test logging errors."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        error = ValueError("Test error")

        # Should not raise
        logger.log_error(error, context="Testing error logging")

    def test_log_completion(self, temp_dir):
        """Test logging training completion."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        summary = {
            "iterations_completed": 10,
            "total_samples": 1000,
            "best_accuracy": 0.92,
            "stopped_early": False,
        }

        # Should not raise
        logger.log_completion(summary)

    def test_get_metrics_history(self, temp_dir):
        """Test getting metrics history."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        logger.log_scalar("loss", 0.5, step=1)
        logger.log_scalar("loss", 0.4, step=2)

        history = logger.get_metrics_history()

        assert "loss" in history
        assert history["loss"] == [0.5, 0.4]

    def test_close(self, temp_dir):
        """Test closing logger."""
        config = LoggingConfig(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)
        logger = AutoTrainLogger(config=config)

        # Should not raise
        logger.close()


class TestGetLogger:
    """Tests for get_logger convenience function."""

    def test_get_logger_default(self, temp_dir):
        """Test get_logger with defaults."""
        logger = get_logger(log_dir=str(temp_dir), enable_tensorboard=False, enable_wandb=False)

        assert isinstance(logger, AutoTrainLogger)
        assert logger.name == "autotrain"


__all__ = ["TestLoggingConfig", "TestAutoTrainLogger", "TestGetLogger"]
