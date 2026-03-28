"""Tests for CPT dataset."""

import json
import tempfile
from pathlib import Path

import pytest

from autotrain.datasets.cpt import CPTDataset, CPTDatasetConfig


class TestCPTDatasetConfig:
    def test_config_defaults(self):
        config = CPTDatasetConfig()
        assert config.name == "cpt_dataset"
        assert config.text_key == "text"
        assert config.max_samples is None

    def test_config_custom(self):
        config = CPTDatasetConfig(name="custom", text_key="content")
        assert config.name == "custom"
        assert config.text_key == "content"


class TestCPTDataset:
    def test_dataset_creation(self):
        dataset = CPTDataset(name="test")
        assert dataset.name == "test"
        assert dataset.is_empty

    def test_add_text(self):
        dataset = CPTDataset()
        dataset.add_text("This is a test document.")
        assert dataset.sample_count == 1
        assert dataset[0].input_data == "This is a test document."

    def test_add_texts_list(self):
        dataset = CPTDataset()
        dataset.add_texts(["doc1", "doc2", "doc3"])
        assert dataset.sample_count == 3

    def test_add_texts_dicts(self):
        dataset = CPTDataset()
        dataset.add_texts([{"text": "doc1"}, {"text": "doc2"}])
        assert dataset.sample_count == 2


class TestCPTDatasetLoad:
    def test_from_list(self):
        texts = ["text1", "text2", "text3"]
        dataset = CPTDataset.from_list(texts, name="test")
        assert dataset.sample_count == 3
        assert dataset[0].input_data == "text1"

    def test_from_list_with_metadata(self):
        texts = ["text1", "text2"]
        metadata = [{"source": "a"}, {"source": "b"}]
        dataset = CPTDataset.from_list(texts, metadata=metadata)
        assert dataset[0].metadata["source"] == "a"
        assert dataset[1].metadata["source"] == "b"


class TestCPTDatasetSave:
    def test_save_jsonl(self, temp_dir):
        dataset = CPTDataset.from_list(["doc1", "doc2"], name="test")
        path = temp_dir / "test.jsonl"
        dataset.save(path, format="jsonl")
        assert path.exists()

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2


class TestCPTDatasetStatistics:
    def test_statistics_empty(self):
        dataset = CPTDataset()
        stats = dataset.get_statistics()
        assert stats["total_samples"] == 0

    def test_statistics(self):
        dataset = CPTDataset.from_list(["hello world", "test document"])
        stats = dataset.get_statistics()
        assert stats["total_samples"] == 2
        assert stats["total_words"] > 0


class TestCPTDatasetIntegration:
    def test_to_list(self):
        dataset = CPTDataset.from_list(["doc1", "doc2"])
        texts = dataset.to_list()
        assert texts == ["doc1", "doc2"]
