"""Tests for CheckpointManager class."""

import json
from unittest.mock import MagicMock

import pytest
from autotrain import CheckpointInfo, CheckpointManager


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_checkpoint_info_creation(self):
        """Test CheckpointInfo creation."""
        info = CheckpointInfo(
            checkpoint_id="ckpt_1",
            iteration=1,
            timestamp="2024-01-01T00:00:00",
            model_name="test-model",
            training_samples_count=100,
            benchmark_accuracy=0.85,
        )

        assert info.checkpoint_id == "ckpt_1"
        assert info.iteration == 1
        assert info.benchmark_accuracy == 0.85

    def test_checkpoint_info_to_dict(self):
        """Test CheckpointInfo.to_dict."""
        info = CheckpointInfo(
            checkpoint_id="ckpt_1",
            iteration=1,
            timestamp="2024-01-01T00:00:00",
            model_name="test-model",
            training_samples_count=100,
            benchmark_accuracy=0.85,
            metadata={"key": "value"},
        )

        data = info.to_dict()

        assert data["checkpoint_id"] == "ckpt_1"
        assert data["metadata"]["key"] == "value"

    def test_checkpoint_info_from_dict(self):
        """Test CheckpointInfo.from_dict."""
        data = {
            "checkpoint_id": "ckpt_1",
            "iteration": 1,
            "timestamp": "2024-01-01T00:00:00",
            "model_name": "test-model",
            "training_samples_count": 100,
            "benchmark_accuracy": 0.85,
            "metadata": {"key": "value"},
        }

        info = CheckpointInfo.from_dict(data)

        assert info.checkpoint_id == "ckpt_1"
        assert info.metadata["key"] == "value"

    def test_checkpoint_info_default_metadata(self):
        """Test CheckpointInfo default metadata."""
        info = CheckpointInfo(
            checkpoint_id="ckpt_1",
            iteration=1,
            timestamp="2024-01-01T00:00:00",
            model_name="test-model",
            training_samples_count=100,
        )

        assert info.metadata == {}


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_manager_init_default(self):
        """Test CheckpointManager initialization with defaults."""
        manager = CheckpointManager()

        assert manager.checkpoint_dir.name == "checkpoints"
        assert manager.max_checkpoints == 5
        assert manager.keep_best is True
        assert manager._checkpoints == []

    def test_manager_init_custom(self, temp_dir):
        """Test CheckpointManager initialization with custom params."""
        checkpoint_dir = temp_dir / "custom_checkpoints"

        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            max_checkpoints=3,
            keep_best=False,
        )

        assert manager.checkpoint_dir == checkpoint_dir
        assert manager.max_checkpoints == 3
        assert manager.keep_best is False

    def test_manager_creates_directory(self, temp_dir):
        """Test that CheckpointManager creates checkpoint directory."""
        checkpoint_dir = temp_dir / "new_checkpoints"

        CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_save_checkpoint(self, temp_dir):
        """Test saving a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.inference_config.temperature = 0.7
        mock_model.prompts = MagicMock()

        info = manager.save(
            model=mock_model,
            iteration=1,
            benchmark_accuracy=0.85,
            metadata={"test": True},
        )

        assert info.iteration == 1
        assert info.benchmark_accuracy == 0.85
        assert len(manager._checkpoints) == 1

    def test_save_creates_checkpoint_directory(self, temp_dir):
        """Test that save creates checkpoint subdirectory."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        checkpoint_path = temp_dir / info.checkpoint_id
        assert checkpoint_path.exists()

    def test_save_saves_info_file(self, temp_dir):
        """Test that save creates checkpoint_info.json."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        info_path = temp_dir / info.checkpoint_id / "checkpoint_info.json"
        assert info_path.exists()

        with open(info_path) as f:
            data = json.load(f)

        assert data["iteration"] == 1

    def test_save_saves_training_data(self, temp_dir):
        """Test that save creates training data files."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = [{"input": "test", "output": "result"}]
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        data_path = temp_dir / info.checkpoint_id / "training_data"
        assert data_path.exists()

    def test_save_saves_config(self, temp_dir):
        """Test that save creates config file."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = True
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.inference_config.temperature = 0.7
        mock_model.inference_config.max_tokens = 1024
        mock_model.inference_config.top_p = 0.9
        mock_model.inference_config.frequency_penalty = 0.0
        mock_model.inference_config.presence_penalty = 0.0
        mock_model.prompts = MagicMock()
        mock_model.prompts.producer = "producer"
        mock_model.prompts.solver = "solver"
        mock_model.prompts.splitter = "splitter"
        mock_model.prompts.reviewer = "reviewer"
        mock_model.prompts.checker = "checker"

        info = manager.save(model=mock_model, iteration=1)

        config_path = temp_dir / info.checkpoint_id / "config" / "config.json"
        assert config_path.exists()

    def test_save_updates_best_checkpoint(self, temp_dir):
        """Test that save updates best checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        # Save first checkpoint with lower accuracy
        manager.save(
            model=mock_model,
            iteration=1,
            benchmark_accuracy=0.7,
        )

        # Save second checkpoint with higher accuracy
        info2 = manager.save(
            model=mock_model,
            iteration=2,
            benchmark_accuracy=0.9,
        )

        assert manager._best_checkpoint_id == info2.checkpoint_id

    def test_save_index_file(self, temp_dir):
        """Test that save creates/updates index file."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1)

        index_path = temp_dir / "checkpoint_index.json"
        assert index_path.exists()

    def test_load_checkpoint(self, temp_dir):
        """Test loading a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        # Create new model for loading
        mock_model2 = MagicMock()
        mock_model2._samples = []
        mock_model2._training_data = []

        loaded_info = manager.load(
            model=mock_model2,
            checkpoint_id=info.checkpoint_id,
        )

        assert loaded_info.iteration == info.iteration

    def test_load_latest_checkpoint(self, temp_dir):
        """Test loading latest checkpoint when no ID specified."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1)
        manager.save(model=mock_model, iteration=2)
        manager.save(model=mock_model, iteration=3)

        mock_model2 = MagicMock()
        mock_model2._samples = []
        mock_model2._training_data = []

        # Should load latest (iteration 3)
        loaded_info = manager.load(model=mock_model2)

        assert loaded_info.iteration == 3

    def test_load_by_iteration(self, temp_dir):
        """Test loading checkpoint by iteration."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1)
        manager.save(model=mock_model, iteration=2)
        manager.save(model=mock_model, iteration=3)

        mock_model2 = MagicMock()
        mock_model2._samples = []
        mock_model2._training_data = []

        loaded_info = manager.load(model=mock_model2, iteration=2)

        assert loaded_info.iteration == 2

    def test_load_not_found(self, temp_dir):
        """Test loading non-existent checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model._samples = []
        mock_model._training_data = []

        with pytest.raises(ValueError, match="No checkpoints available"):
            manager.load(model=mock_model)

    def test_load_iteration_not_found(self, temp_dir):
        """Test loading checkpoint for non-existent iteration."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1)

        mock_model2 = MagicMock()
        mock_model2._samples = []
        mock_model2._training_data = []

        with pytest.raises(ValueError, match="No checkpoint found for iteration"):
            manager.load(model=mock_model2, iteration=999)

    def test_list_checkpoints(self, temp_dir):
        """Test listing checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1)
        manager.save(model=mock_model, iteration=2)
        manager.save(model=mock_model, iteration=3)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3

    def test_get_checkpoint_info(self, temp_dir):
        """Test getting specific checkpoint info."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        retrieved = manager.get_checkpoint_info(info.checkpoint_id)

        assert retrieved is not None
        assert retrieved.iteration == 1

    def test_get_checkpoint_info_not_found(self, temp_dir):
        """Test getting non-existent checkpoint info."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        retrieved = manager.get_checkpoint_info("nonexistent")

        assert retrieved is None

    def test_get_best_checkpoint(self, temp_dir):
        """Test getting best checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1, benchmark_accuracy=0.7)
        manager.save(model=mock_model, iteration=2, benchmark_accuracy=0.9)
        manager.save(model=mock_model, iteration=3, benchmark_accuracy=0.8)

        best = manager.get_best_checkpoint()

        assert best is not None
        assert best.iteration == 2
        assert best.benchmark_accuracy == 0.9

    def test_restore_best(self, temp_dir):
        """Test restoring best checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        manager.save(model=mock_model, iteration=1, benchmark_accuracy=0.7)
        manager.save(model=mock_model, iteration=2, benchmark_accuracy=0.9)

        mock_model2 = MagicMock()
        mock_model2._samples = []
        mock_model2._training_data = []

        restored = manager.restore_best(mock_model2)

        assert restored is not None
        assert restored.iteration == 2

    def test_restore_best_no_checkpoints(self, temp_dir):
        """Test restoring best when no checkpoints exist."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model._samples = []
        mock_model._training_data = []

        result = manager.restore_best(mock_model)

        assert result is None

    def test_export_checkpoint(self, temp_dir):
        """Test exporting checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        export_path = temp_dir / "exported"
        manager.export_checkpoint(info.checkpoint_id, str(export_path))

        assert export_path.exists()

    def test_export_checkpoint_not_found(self, temp_dir):
        """Test exporting non-existent checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            manager.export_checkpoint("nonexistent", str(temp_dir / "exported"))

    def test_cleanup_old_checkpoints(self, temp_dir):
        """Test cleanup of old checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            max_checkpoints=2,
            keep_best=True,
        )

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        # Save 4 checkpoints with increasing accuracy
        for i in range(4):
            manager.save(
                model=mock_model,
                iteration=i,
                benchmark_accuracy=0.5 + i * 0.1,
            )

        # Should keep: best (iteration 3), and max_checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 3  # best + max_checkpoints

    def test_delete_checkpoint(self, temp_dir):
        """Test deleting a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        checkpoint_path = temp_dir / info.checkpoint_id
        assert checkpoint_path.exists()

        manager._delete_checkpoint(info.checkpoint_id)

        assert not checkpoint_path.exists()

    def test_load_checkpoint_restores_training_data(self, temp_dir):
        """Test that loading checkpoint restores training data."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 2
        mock_model.enable_checker = False
        mock_model._samples = [MagicMock(input_data="input1", output_data="output1", metadata={})]
        mock_model._training_data = [{"input": "test", "output": "result"}]
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        # Create new model for loading
        mock_model2 = MagicMock()
        mock_model2._samples = []
        mock_model2._training_data = []

        manager.load(model=mock_model2, checkpoint_id=info.checkpoint_id)

        assert len(mock_model2._samples) == 1
        assert len(mock_model2._training_data) == 1

    def test_load_checkpoint_restores_config(self, temp_dir):
        """Test that loading checkpoint restores config."""
        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        mock_model.sample_multiplier = 3
        mock_model.enable_checker = True
        mock_model._samples = []
        mock_model._training_data = []
        mock_model._fast_model = None
        mock_model.inference_config = MagicMock()
        mock_model.prompts = MagicMock()

        info = manager.save(model=mock_model, iteration=1)

        # Create new model for loading
        mock_model2 = MagicMock()
        mock_model2.model_name = "different-model"
        mock_model2.sample_multiplier = 1
        mock_model2.enable_checker = False
        mock_model2._samples = []
        mock_model2._training_data = []

        manager.load(model=mock_model2, checkpoint_id=info.checkpoint_id)

        # Config should be restored
        assert mock_model2.sample_multiplier == 3
        assert mock_model2.enable_checker is True
