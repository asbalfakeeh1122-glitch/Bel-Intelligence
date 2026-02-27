import os
import time

import psutil
import pytest
import torch

from app.core.nlp_pipeline import nlp_service


@pytest.mark.asyncio
async def test_resource_footprint():
    """Elite Benchmarking: Verify RAM and GPU usage under load."""
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / (1024 * 1024)

    start_time = time.time()
    # Mock heavy analysis
    text = "The quick brown fox jumps over the lazy dog. " * 100
    await nlp_service.analyze(text, ["Nature", "Action"])

    end_time = time.time()
    final_mem = process.memory_info().rss / (1024 * 1024)

    latency = end_time - start_time
    mem_delta = final_mem - initial_mem

    print(f"\n[BENCHMARK] Latency: {latency:.2f}s")
    print(f"[BENCHMARK] Memory Usage Increase: {mem_delta:.2f} MB")

    # Assertions for World-Tier Performance
    assert latency < 5.0  # Should be fast for this size
    assert mem_delta < 2000  # Should not leak memory significantly during single call


def test_gpu_utilization():
    """Verify GPU is used if available."""
    if torch.cuda.is_available():
        # Check if any model is on 'cuda'
        on_cuda = False
        if hasattr(nlp_service, "sentiment_model_raw"):
            on_cuda = next(nlp_service.sentiment_model_raw.parameters()).is_cuda
        assert on_cuda, "GPU available but model not loaded on CUDA"
    else:
        pytest.skip("No GPU available for testing")
