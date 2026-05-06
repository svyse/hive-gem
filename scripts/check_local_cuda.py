#!/usr/bin/env python3
"""Check whether the backend Python environment can use the RTX/CUDA GPU."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as e:
        print(f"torch import failed: {type(e).__name__}: {e}")
        print("Install a CUDA-enabled PyTorch build in this same backend environment.")
        return 2

    print(f"torch: {getattr(torch, '__version__', 'unknown')}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA is not visible to PyTorch. The local model and trainer will use CPU.")
        print("For RTX 3060 on Windows, install PyTorch CUDA in this environment, for example:")
        print("  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return 1

    print(f"cuda_device_count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = int(getattr(props, 'total_memory', 0) or 0) // (1024 * 1024)
        allocated = int(torch.cuda.memory_allocated(i)) // (1024 * 1024)
        reserved = int(torch.cuda.memory_reserved(i)) // (1024 * 1024)
        print(f"cuda:{i}: {props.name} | total={total} MiB | allocated={allocated} MiB | reserved={reserved} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
