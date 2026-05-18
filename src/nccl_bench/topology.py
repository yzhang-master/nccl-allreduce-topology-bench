import itertools
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    gpus: tuple[str, ...]
    nvlink_pairs: tuple[tuple[str, str], ...] = ()
    nics: tuple[str, ...] = ()
    nic_affinities: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.gpus or len(self.gpus) != len(set(self.gpus)):
            raise ValueError("gpus must be non-empty and unique")
        if len(self.nics) != len(set(self.nics)):
            raise ValueError("NIC names must be unique")
        for left, right in self.nvlink_pairs:
            if left == right or left not in self.gpus or right not in self.gpus:
                raise ValueError("NVLink pairs must connect distinct known GPUs")
        for gpu, nic in self.nic_affinities:
            if gpu not in self.gpus or nic not in self.nics:
                raise ValueError("NIC affinity must reference known devices")

    def nvlink_adjacency(self) -> dict[str, set[str]]:
        adjacency = {gpu: set() for gpu in self.gpus}
        for left, right in self.nvlink_pairs:
            adjacency[left].add(right)
            adjacency[right].add(left)
        return adjacency


def parse_nvidia_smi_topology(text: str) -> Topology:
    """Parse GPU/NIC links from `nvidia-smi topo -m`, including its NIC legend."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = next((line.split() for line in lines if line.startswith("GPU0 ")), None)
    if header is None:
        raise ValueError("missing topology header")
    columns = []
    for token in header:
        if token in {"CPU", "NUMA"}:
            break
        columns.append(token)
    legend = dict(re.findall(r"^(NIC\d+):\s+(\S+)", text, flags=re.MULTILINE))
    columns = [legend.get(column, column) for column in columns]
    gpus = tuple(column for column in columns if re.fullmatch(r"GPU\d+", column))
    nics = tuple(column for column in columns if column not in gpus)
    nvlink_pairs = set()
    nic_affinities = set()
    for line in lines:
        fields = line.split()
        if not fields or fields[0] not in gpus:
            continue
        gpu = fields[0]
        for column, value in zip(columns, fields[1:], strict=False):
            if column in gpus and value.startswith("NV"):
                nvlink_pairs.add(tuple(sorted((gpu, column))))
            if column in nics and value in {"PIX", "PXB", "PHB", "NODE"}:
                nic_affinities.add((gpu, column))
    return Topology(
        gpus=gpus,
        nvlink_pairs=tuple(sorted(nvlink_pairs)),
        nics=nics,
        nic_affinities=tuple(sorted(nic_affinities)),
    )


def nvlink_edge_count(order: tuple[str, ...], topology: Topology) -> int:
    if len(order) != len(topology.gpus) or set(order) != set(topology.gpus):
        raise ValueError("ring order must contain every GPU exactly once")
    if len(order) <= 1:
        return 0
    pairs = set(topology.nvlink_pairs)
    count = 0
    for index, gpu in enumerate(order):
        other = order[(index + 1) % len(order)]
        if tuple(sorted((gpu, other))) in pairs:
            count += 1
    return count


def best_nvlink_ring(topology: Topology) -> tuple[str, ...]:
    """Maximize NVLink cycle edges exactly through 8 GPUs, greedily above that."""
    gpus = tuple(sorted(topology.gpus, key=lambda name: int(name.removeprefix("GPU"))))
    if len(gpus) <= 1:
        return gpus
    adjacency = topology.nvlink_adjacency()
    if len(gpus) <= 8:
        candidates = ((gpus[0], *rest) for rest in itertools.permutations(gpus[1:]))
        return max(candidates, key=lambda order: (nvlink_edge_count(order, topology), order))
    remaining = set(gpus[1:])
    order = [gpus[0]]
    while remaining:
        current = order[-1]
        following = min(remaining, key=lambda gpu: (gpu not in adjacency[current], gpu))
        order.append(following)
        remaining.remove(following)
    return tuple(order)


def recommend_nccl_environment(
    topology: Topology,
    *,
    message_bytes: int,
    socket_ifname: str | None = None,
) -> dict[str, str]:
    """Deterministic starting point for tuning, not a claim of optimal performance."""
    if message_bytes <= 0:
        raise ValueError("message_bytes must be positive")
    order = best_nvlink_ring(topology)
    complete_cycle = len(order) > 1 and nvlink_edge_count(order, topology) == len(order)
    environment = {
        "NCCL_ALGO": "Ring" if complete_cycle else "Tree",
        "NCCL_PROTO": "LL" if message_bytes <= 256 * 1024 else "Simple",
    }
    if topology.nics:
        environment["NCCL_IB_HCA"] = ",".join(topology.nics)
    if socket_ifname:
        environment["NCCL_SOCKET_IFNAME"] = socket_ifname
    return environment


def ring_allreduce_bytes_per_rank(message_bytes: int, world_size: int) -> float:
    if message_bytes <= 0 or world_size <= 0:
        raise ValueError("message size and world size must be positive")
    return 2 * (world_size - 1) * message_bytes / world_size


def bus_bandwidth_gbps(message_bytes: int, world_size: int, elapsed_seconds: float) -> float:
    """NCCL bus bandwidth: algorithmic bandwidth times 2*(world_size-1)/world_size."""
    if elapsed_seconds <= 0:
        raise ValueError("elapsed time must be positive")
    algorithmic = message_bytes / elapsed_seconds / 1e9
    return algorithmic * 2 * (world_size - 1) / world_size
