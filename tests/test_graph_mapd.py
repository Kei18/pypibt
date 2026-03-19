"""Tests for the graph-based MAPD (LIF-graph) solver and its utilities."""
import os
import pytest

from pypibt import (
    GraphMAPD,
    GraphTask,
    GraphMAPDInstance,
    is_valid_graph_mapd_solution,
    save_graph_mapd_solution,
    parse_lif,
    parse_transport_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "graph-mapd")
LIF_FILE = os.path.join(_ASSETS, "lif_example.json")
TRANSPORT_FILE = os.path.join(_ASSETS, "transport_example.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "local")


def _make_solver(num_agents: int = 2, task_num: int = 10, seed: int = 0,
                 task_mode: str = "per_agent") -> GraphMAPD:
    return GraphMAPD(
        LIF_FILE, TRANSPORT_FILE,
        num_agents=num_agents, task_num=task_num, seed=seed, task_mode=task_mode,
    )


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

def test_graphmapd_init_basic():
    """GraphMAPD initialises correctly and exposes the expected attributes."""
    gm = _make_solver(num_agents=2, task_num=10)
    assert gm.N == 2
    assert len(gm.starts) == 2
    # all start nodes are valid indices
    assert all(0 <= s < gm.graph.num_nodes for s in gm.starts)
    # no two agents share the same start node
    assert len(set(gm.starts)) == gm.N
    assert gm.instance.task_num == 10
    assert gm.solution == []
    assert gm.completed_tasks == []


def test_graphmapd_init_invalid_task_mode():
    """Unsupported task_mode raises ValueError at construction."""
    with pytest.raises(ValueError, match="task_mode"):
        GraphMAPD(LIF_FILE, TRANSPORT_FILE, num_agents=2, task_num=10,
                  task_mode="invalid")


def test_graphmapd_init_global_task_mode():
    """Global task mode creates a non-empty global queue."""
    gm = _make_solver(num_agents=2, task_num=10, task_mode="global")
    assert len(gm.instance.global_task_queue) == 10
    assert all(len(q) == 0 for q in gm.instance.agent_task_queues)


def test_graphmapd_init_per_agent_task_mode():
    """Per-agent task mode distributes tasks across individual queues."""
    gm = _make_solver(num_agents=2, task_num=10, task_mode="per_agent")
    total = sum(len(q) for q in gm.instance.agent_task_queues)
    assert total == 10
    assert gm.instance.global_task_queue == []


# ---------------------------------------------------------------------------
# Run / solution tests
# ---------------------------------------------------------------------------

def test_graphmapd_run_returns_solution():
    """run() returns a non-empty solution and populates stats."""
    gm = _make_solver(num_agents=2, task_num=20)
    solution = gm.run(max_timestep=500)
    assert len(solution) > 0
    # first row must equal the initial positions
    assert list(solution[0]) == list(gm.starts)
    # every config must have exactly N entries
    for config in solution:
        assert len(config) == gm.N
    assert gm.stats, "stats dict should be populated after run()"


def test_graphmapd_run_valid_solution():
    """The solution produced by run() must pass the built-in validator."""
    gm = _make_solver(num_agents=3, task_num=30, seed=42)
    gm.run(max_timestep=1000)
    assert is_valid_graph_mapd_solution(
        gm.graph, gm.starts, gm.solution, gm.completed_tasks, verbose=True
    )


def test_graphmapd_run_valid_solution_global_mode():
    """Validator must pass with global task mode too."""
    gm = _make_solver(num_agents=4, task_num=40, seed=7, task_mode="global")
    gm.run(max_timestep=2000)
    assert is_valid_graph_mapd_solution(
        gm.graph, gm.starts, gm.solution, gm.completed_tasks, verbose=True
    )


def test_graphmapd_run_respects_max_timestep():
    """run() must not produce more than max_timestep+1 configurations."""
    max_ts = 50
    gm = _make_solver(num_agents=2, task_num=200)
    gm.run(max_timestep=max_ts)
    assert len(gm.solution) <= max_ts + 1


def test_graphmapd_run_completes_tasks():
    """At least some tasks are completed during a normal run."""
    gm = _make_solver(num_agents=4, task_num=20, seed=1)
    gm.run(max_timestep=2000)
    assert len(gm.completed_tasks) > 0


def test_graphmapd_completed_tasks_structure():
    """Each completed task has a valid agent id and a consistent task object."""
    gm = _make_solver(num_agents=3, task_num=30, seed=5)
    gm.run(max_timestep=2000)
    for agent_id, task in gm.completed_tasks:
        assert 0 <= agent_id < gm.N
        assert isinstance(task, GraphTask)
        assert task.timestep_finished is not None
        assert task.timestep_finished >= task.timestep_appear
        assert 0 <= task.node_pickup < gm.graph.num_nodes
        assert 0 <= task.node_delivery < gm.graph.num_nodes


def test_graphmapd_deterministic():
    """Two runs with the same seed must produce identical solutions."""
    gm1 = _make_solver(num_agents=3, task_num=30, seed=99)
    gm2 = _make_solver(num_agents=3, task_num=30, seed=99)
    sol1 = gm1.run(max_timestep=500)
    sol2 = gm2.run(max_timestep=500)
    assert sol1 == sol2
    assert len(gm1.completed_tasks) == len(gm2.completed_tasks)


def test_graphmapd_different_seeds_differ():
    """Two runs with different seeds should (almost certainly) differ."""
    gm1 = _make_solver(num_agents=4, task_num=40, seed=0)
    gm2 = _make_solver(num_agents=4, task_num=40, seed=1)
    gm1.run(max_timestep=500)
    gm2.run(max_timestep=500)
    assert gm1.starts != gm2.starts or gm1.solution != gm2.solution


def test_graphmapd_stats_keys():
    """stats contains the expected keys after run()."""
    expected = {
        "tasks_completed", "tasks_requested", "target_reached",
        "num_agents", "timesteps", "makespan_s", "flowtime_s",
        "congestion_rate_s", "total_distance_m", "throughput_tasks_per_s",
    }
    gm = _make_solver(num_agents=2, task_num=10)
    gm.run(max_timestep=500)
    for key in expected:
        assert key in gm.stats, f"Missing stats key: {key}"


def test_graphmapd_stop_before():
    """stop_before causes the run to halt before all tasks are done."""
    task_num = 20
    stop_before = 5
    gm = _make_solver(num_agents=3, task_num=task_num)
    gm.run(max_timestep=5000, stop_before=stop_before)
    # completed count <= task_num - stop_before
    assert len(gm.instance.tasks_completed) <= task_num - stop_before + gm.N


# ---------------------------------------------------------------------------
# Validator edge-case tests
# ---------------------------------------------------------------------------

def test_validator_empty_solution():
    """Validator returns False for an empty solution."""
    gm = _make_solver()
    assert not is_valid_graph_mapd_solution(gm.graph, gm.starts, [], [], verbose=False)


def test_validator_wrong_starts():
    """Validator returns False when the first config doesn't match starts."""
    gm = _make_solver(num_agents=2, task_num=10)
    gm.run(max_timestep=200)
    # corrupt the starts so they don't match solution[0]
    bad_starts = [(s + 1) % gm.graph.num_nodes for s in gm.starts]
    assert not is_valid_graph_mapd_solution(
        gm.graph, bad_starts, gm.solution, [], verbose=False
    )


def test_validator_vertex_collision():
    """Validator detects vertex collisions (two agents at same node)."""
    gm = _make_solver(num_agents=2, task_num=5)
    gm.run(max_timestep=100)
    # Build a corrupted solution: put both agents on node 0 at t=1
    good_sol = [list(c) for c in gm.solution]
    if len(good_sol) > 1:
        good_sol[1][0] = good_sol[1][1]  # force vertex collision
    assert not is_valid_graph_mapd_solution(
        gm.graph, gm.starts, good_sol, [], verbose=False
    )


def test_validator_invalid_move():
    """Validator detects teleportation (agent moves to non-adjacent node)."""
    gm = _make_solver(num_agents=2, task_num=5, seed=0)
    gm.run(max_timestep=100)
    good_sol = [list(c) for c in gm.solution]
    if len(good_sol) > 1:
        # find a node that is NOT adjacent to agent 0's position at t=0
        adj = gm.graph.adjacency
        current = good_sol[0][0]
        non_adjacent = next(
            n for n in range(gm.graph.num_nodes)
            if n != current and n not in adj.get(current, [])
        )
        good_sol[1][0] = non_adjacent  # teleport
    assert not is_valid_graph_mapd_solution(
        gm.graph, gm.starts, good_sol, [], verbose=False
    )


# ---------------------------------------------------------------------------
# Save solution test
# ---------------------------------------------------------------------------

def test_save_graph_mapd_solution():
    """save_graph_mapd_solution creates a non-empty file."""
    gm = _make_solver(num_agents=2, task_num=10)
    gm.run(max_timestep=300)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "test_graph_mapd_output.txt")
    save_graph_mapd_solution(
        gm.graph, gm.solution, out,
        completed_tasks=gm.completed_tasks,
        stats=gm.stats,
    )
    assert os.path.exists(out)
    with open(out) as f:
        content = f.read()
    assert "agents=" in content


# ---------------------------------------------------------------------------
# GraphMAPDInstance unit tests
# ---------------------------------------------------------------------------

def test_graphmapd_instance_task_counts():
    """Task queues sum to exactly task_num in per_agent mode."""
    graph = parse_lif(LIF_FILE)
    transport = parse_transport_matrix(TRANSPORT_FILE)
    inst = GraphMAPDInstance(graph, transport, task_num=15, num_agents=3, seed=0,
                             task_mode="per_agent")
    total = sum(len(q) for q in inst.agent_task_queues)
    assert total == 15


def test_graphmapd_instance_global_queue_length():
    """Global queue has exactly task_num tasks in global mode."""
    graph = parse_lif(LIF_FILE)
    transport = parse_transport_matrix(TRANSPORT_FILE)
    inst = GraphMAPDInstance(graph, transport, task_num=17, num_agents=3, seed=0,
                             task_mode="global")
    assert len(inst.global_task_queue) == 17


def test_graphmapd_instance_invalid_mode():
    """GraphMAPDInstance raises ValueError for unknown task_mode."""
    graph = parse_lif(LIF_FILE)
    transport = parse_transport_matrix(TRANSPORT_FILE)
    with pytest.raises(ValueError, match="task_mode"):
        GraphMAPDInstance(graph, transport, task_num=10, num_agents=2, task_mode="bad")
