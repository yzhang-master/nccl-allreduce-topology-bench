import pytest

from nccl_bench.topology import (
    Topology,
    best_nvlink_ring,
    bus_bandwidth_gbps,
    nvlink_edge_count,
    parse_nvidia_smi_topology,
    recommend_nccl_environment,
    ring_allreduce_bytes_per_rank,
)

TOPO_OUTPUT = """
        GPU0    GPU1    GPU2    GPU3    NIC0    CPU Affinity    NUMA Affinity
GPU0     X       NV2     SYS     SYS     PXB     0-15           0
GPU1     NV2     X       NV2     SYS     SYS     0-15           0
GPU2     SYS     NV2     X       NV2     SYS     16-31          1
GPU3     SYS     SYS     NV2     X       PHB     16-31          1
NIC0     PXB     SYS     SYS     PHB     X
NIC0: mlx5_0
"""


def test_parse_nvlink_and_infiniband_topology():
    topology = parse_nvidia_smi_topology(TOPO_OUTPUT)
    assert topology.gpus == ("GPU0", "GPU1", "GPU2", "GPU3")
    assert topology.nvlink_pairs == (
        ("GPU0", "GPU1"),
        ("GPU1", "GPU2"),
        ("GPU2", "GPU3"),
    )
    assert topology.nics == ("mlx5_0",)
    assert topology.nic_affinities == (("GPU0", "mlx5_0"), ("GPU3", "mlx5_0"))


def test_ring_selection_uses_topology_and_rejects_invalid_order():
    topology = parse_nvidia_smi_topology(TOPO_OUTPUT)
    order = best_nvlink_ring(topology)
    assert nvlink_edge_count(order, topology) == 3
    closed = Topology(
        gpus=topology.gpus,
        nvlink_pairs=topology.nvlink_pairs + (("GPU0", "GPU3"),),
    )
    best = best_nvlink_ring(closed)
    assert nvlink_edge_count(best, closed) == 4
    with pytest.raises(ValueError):
        nvlink_edge_count(("GPU0", "GPU0", "GPU2", "GPU3"), topology)


def test_nccl_environment_depends_on_cycle_message_size_and_ib():
    chain = parse_nvidia_smi_topology(TOPO_OUTPUT)
    assert recommend_nccl_environment(chain, message_bytes=1024) == {
        "NCCL_ALGO": "Tree",
        "NCCL_PROTO": "LL",
        "NCCL_IB_HCA": "mlx5_0",
    }
    closed = Topology(
        gpus=chain.gpus,
        nvlink_pairs=chain.nvlink_pairs + (("GPU0", "GPU3"),),
        nics=chain.nics,
    )
    assert recommend_nccl_environment(
        closed, message_bytes=1024 * 1024, socket_ifname="ib0"
    ) == {
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
        "NCCL_IB_HCA": "mlx5_0",
        "NCCL_SOCKET_IFNAME": "ib0",
    }


def test_ring_traffic_and_bus_bandwidth_match_nccl_formula():
    assert ring_allreduce_bytes_per_rank(8_000_000, 4) == 12_000_000
    assert ring_allreduce_bytes_per_rank(8_000_000, 1) == 0
    assert bus_bandwidth_gbps(8_000_000, 4, 0.01) == pytest.approx(1.2)
    assert bus_bandwidth_gbps(8_000_000, 1, 0.01) == 0
    with pytest.raises(ValueError):
        bus_bandwidth_gbps(8_000_000, 4, 0)
