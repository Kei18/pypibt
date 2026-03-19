"""LIF format parser and directed-graph utilities for Graph MAPD.

Parses LIF (Layout Interchange Format) JSON files into a directed graph
(LifGraph) for use with the GraphMAPD solver. Edge lengths are derived from
the Euclidean distance between node positions and accumulated during simulation
for throughput normalisation.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

# Public type aliases
NodeId: TypeAlias = str   # string node identifier from the LIF file (e.g. "N1")
NodeIdx: TypeAlias = int  # integer node index used for internal array operations


@dataclass
class LifGraph:
    """Directed graph representation of a single LIF layout.

    Uses integer node indices internally so that occupation tables and distance
    caches can be plain numpy arrays.  String node IDs are preserved for output.

    Attributes:
        node_ids: Maps NodeIdx -> original string node ID (e.g. "N1").
        node_index: Maps string node ID -> NodeIdx.
        node_positions: NodeIdx -> (x, y) real-world coordinates (metres).
        adjacency: Forward adjacency list  NodeIdx -> list[NodeIdx].
        rev_adjacency: Reverse adjacency list NodeIdx -> list[NodeIdx].
            Used by GraphDistTable to compute dist(v -> goal) on directed graphs.
        edge_lengths: (from_idx, to_idx) -> Euclidean length.
        station_nodes: stationId -> list of interaction NodeIdx values.
        station_descriptions: stationId -> description string (e.g. "INPUT").
    """

    node_ids: list[NodeId]
    node_index: dict[NodeId, NodeIdx]
    node_positions: dict[NodeIdx, tuple[float, float]]
    adjacency: dict[NodeIdx, list[NodeIdx]]
    rev_adjacency: dict[NodeIdx, list[NodeIdx]]
    edge_lengths: dict[tuple[NodeIdx, NodeIdx], float]
    station_nodes: dict[str, list[NodeIdx]]
    station_descriptions: dict[str, str]

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


def parse_lif(lif_file: str) -> LifGraph:
    """Parse a LIF JSON file into a :class:`LifGraph`.

    Reads the first layout found in the file.  Edge direction is preserved; for
    bidirectional layouts both directions must be listed as separate edges (as
    the standard requires).

    Args:
        lif_file: Path to the LIF JSON file.

    Returns:
        Populated :class:`LifGraph`.

    Raises:
        ValueError: If the file contains no layouts.
        KeyError: If an edge references an unknown node ID.
    """
    with open(lif_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    layouts = data.get("layouts", [])
    if not layouts:
        raise ValueError(f"No layouts found in LIF file: {lif_file}")

    layout = layouts[0]

    # Build node index
    raw_nodes = layout.get("nodes", [])
    node_ids: list[NodeId] = [n["nodeId"] for n in raw_nodes]
    node_index: dict[NodeId, NodeIdx] = {nid: i for i, nid in enumerate(node_ids)}
    node_positions: dict[NodeIdx, tuple[float, float]] = {
        node_index[n["nodeId"]]: (
            float(n["nodePosition"]["x"]),
            float(n["nodePosition"]["y"]),
        )
        for n in raw_nodes
    }

    n = len(node_ids)
    adjacency: dict[NodeIdx, list[NodeIdx]] = {i: [] for i in range(n)}
    rev_adjacency: dict[NodeIdx, list[NodeIdx]] = {i: [] for i in range(n)}
    edge_lengths: dict[tuple[NodeIdx, NodeIdx], float] = {}

    for edge in layout.get("edges", []):
        s = node_index[edge["startNodeId"]]
        e = node_index[edge["endNodeId"]]
        x1, y1 = node_positions[s]
        x2, y2 = node_positions[e]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        adjacency[s].append(e)
        rev_adjacency[e].append(s)
        edge_lengths[(s, e)] = length

    station_nodes: dict[str, list[NodeIdx]] = {}
    station_descriptions: dict[str, str] = {}
    for station in layout.get("stations", []):
        sid = station["stationId"]
        station_nodes[sid] = [
            node_index[nid]
            for nid in station.get("interactionNodeIds", [])
            if nid in node_index
        ]
        station_descriptions[sid] = station.get("stationDescription", "")

    return LifGraph(
        node_ids=node_ids,
        node_index=node_index,
        node_positions=node_positions,
        adjacency=adjacency,
        rev_adjacency=rev_adjacency,
        edge_lengths=edge_lengths,
        station_nodes=station_nodes,
        station_descriptions=station_descriptions,
    )


@dataclass
class TransportMatrix:
    """Weighted material flow between stations.

    Attributes:
        flows: List of (from_station_id, to_station_id, quantity) tuples.
            Only entries with quantity > 0 are included.
        station_ids: All station IDs present in the ``stations`` list of the
            source file (converted to "S{id}" form to match LIF station IDs).
    """

    flows: list[tuple[str, str, float]]
    station_ids: list[str]


def parse_transport_matrix(tm_file: str) -> TransportMatrix:
    """Parse a transport matrix JSON file.

    Station identifiers in the file are numeric (1, 2, …) and are mapped to LIF
    station IDs by prepending ``"S"`` (e.g. ``1`` → ``"S1"``).

    Expected format::

        {
            "stations": [1, 2, 3],
            "materialFlow": [
                {"from": 1, "to": 2, "quantity": 3},
                ...
            ]
        }

    Args:
        tm_file: Path to the transport matrix JSON file.

    Returns:
        :class:`TransportMatrix` containing all non-zero flow entries.
    """
    with open(tm_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    station_ids = [f"S{s}" for s in data.get("stations", [])]
    flows = [
        (f"S{entry['from']}", f"S{entry['to']}", float(entry["quantity"]))
        for entry in data.get("materialFlow", [])
        if float(entry["quantity"]) > 0
    ]
    return TransportMatrix(flows=flows, station_ids=station_ids)


@dataclass
class GraphDistTable:
    """Lazy BFS distance table for a directed graph.

    Computes the minimum number of hops from any node to a fixed *goal* node.
    BFS is performed on the **reverse** adjacency list so that
    ``table[v]`` correctly represents ``dist(v → goal)`` on a directed graph.

    Distances are computed on demand and cached; subsequent queries for already-
    visited nodes are O(1).

    Attributes:
        rev_adjacency: Reverse adjacency (NodeIdx -> predecessor NodeIdxs).
        num_nodes: Total number of nodes in the graph.
        goal: Goal node index.
        Q: BFS frontier (lazy evaluation).
        table: ``table[v] = dist(v → goal)``.  Initialised to ``num_nodes``
            (used as ∞ sentinel, analogous to ``grid.size`` in DistTable).
    """

    rev_adjacency: dict[NodeIdx, list[NodeIdx]]
    num_nodes: int
    goal: NodeIdx
    Q: deque = field(init=False)
    table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.Q = deque([self.goal])
        self.table = np.full(self.num_nodes, self.num_nodes, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: NodeIdx) -> int:
        """Return the minimum hop count from *target* to the goal node.

        Args:
            target: Source node index.

        Returns:
            Hop distance ``dist(target → goal)``.  Returns ``num_nodes`` when
            *target* is out of range or the goal is unreachable from *target*.
        """
        if target < 0 or target >= self.num_nodes:
            return self.num_nodes

        # Already computed
        if int(self.table[target]) < self.num_nodes:
            return int(self.table[target])

        # BFS on reverse graph: a predecessor v of u can reach goal via u
        while self.Q:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in self.rev_adjacency.get(u, []):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.num_nodes
