"""Tests for embedding batch size and CPU thread configuration."""

from unittest.mock import patch

from leann.embedding_compute import (
    _cap_cuda_batch_by_vram,
    _parse_positive_int_env,
    _resolve_adaptive_batch_size,
    _resolve_cpu_thread_count,
)


def test_parse_positive_int_env_default(monkeypatch):
    monkeypatch.delenv("LEANN_TEST_INT", raising=False)
    assert _parse_positive_int_env("LEANN_TEST_INT", 256) == 256


def test_parse_positive_int_env_override(monkeypatch):
    monkeypatch.setenv("LEANN_TEST_INT", "32")
    assert _parse_positive_int_env("LEANN_TEST_INT", 256) == 32


def test_parse_positive_int_env_invalid(monkeypatch):
    monkeypatch.setenv("LEANN_TEST_INT", "not-a-number")
    assert _parse_positive_int_env("LEANN_TEST_INT", 256) == 256


def test_resolve_adaptive_batch_size_cuda(monkeypatch):
    monkeypatch.setenv("LEANN_CUDA_BATCH_SIZE", "64")
    assert _resolve_adaptive_batch_size("cuda", "BAAI/bge-base-en-v1.5") == 64


def test_resolve_adaptive_batch_size_mps_qwen(monkeypatch):
    monkeypatch.delenv("LEANN_MPS_BATCH_SIZE", raising=False)
    assert _resolve_adaptive_batch_size("mps", "Qwen/Qwen3-Embedding-0.6B") == 32


def test_resolve_cpu_threads(monkeypatch):
    monkeypatch.setenv("LEANN_CPU_THREADS", "16")
    assert _resolve_cpu_thread_count() == 16


def test_cap_cuda_batch_by_vram_disabled(monkeypatch):
    monkeypatch.setenv("LEANN_CUDA_AUTO_BATCH", "0")
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.mem_get_info", return_value=(100, 1000)):
            assert _cap_cuda_batch_by_vram(256) == 256


def test_cap_cuda_batch_by_vram_small_gpu(monkeypatch):
    monkeypatch.delenv("LEANN_CUDA_AUTO_BATCH", raising=False)
    # Typical free VRAM on a 4 GiB GPU after loading a base-sized encoder.
    one_gb = 1024**3
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.mem_get_info", return_value=(one_gb, 4 * one_gb)):
            capped = _cap_cuda_batch_by_vram(256, max_length=512)
    assert capped < 256
    assert capped >= 1


def test_cap_cuda_batch_by_vram_four_gb_gpu(monkeypatch):
    """Regression: 4 GiB RTX A1000 reports ~3.2 GiB free; cap should land near 76."""
    monkeypatch.delenv("LEANN_CUDA_AUTO_BATCH", raising=False)
    free_vram = int(3.2 * 1024**3)
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.mem_get_info", return_value=(free_vram, 4 * 1024**3)):
            capped = _cap_cuda_batch_by_vram(256, max_length=512)
    assert capped <= 85
    assert capped >= 1
