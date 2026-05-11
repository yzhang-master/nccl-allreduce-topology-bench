"""Topology-aware collective benchmarking with CPU correctness checks."""

__version__ = "0.1.0"


def scaffold_info() -> dict[str, object]:
    import torch.distributed as dist

    return {
        "package": "nccl_bench",
        "version": __version__,
        "gloo_available": dist.is_gloo_available(),
    }
