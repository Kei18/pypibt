from .mapf_utils import (
    get_grid,
    get_scenario,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
)
from .pibt import PIBT
from .mapd import MAPD
from .mapd_utils import (
    is_dead_end,
    get_valid_positions,
    Task,
    MAPDInstance,
    get_mapd_instance,
    is_valid_mapd_solution,
    save_mapd_solution,
)
from .graph_mapd_utils import (
    LifGraph,
    TransportMatrix,
    GraphDistTable,
    parse_lif,
    parse_transport_matrix,
)
from .graph_mapd import (
    DeadEnd,
    GraphTask,
    GraphAgent,
    GraphMAPDInstance,
    GraphMAPD,
    save_graph_mapd_solution,
    is_valid_graph_mapd_solution,
)

# Backward-compatibility aliases (deprecated)
LifTask = GraphTask
LifAgent = GraphAgent
LifMAPDInstance = GraphMAPDInstance
LifMAPD = GraphMAPD
save_lif_solution = save_graph_mapd_solution
is_valid_lif_mapd_solution = is_valid_graph_mapd_solution

__all__ = [
    "get_grid",
    "get_scenario",
    "is_valid_mapf_solution",
    "save_configs_for_visualizer",
    "is_dead_end",
    "get_valid_positions",
    "PIBT",
    "MAPD",
    "Task",
    "MAPDInstance",
    "get_mapd_instance",
    "is_valid_mapd_solution",
    "save_mapd_solution",
    "LifGraph",
    "TransportMatrix",
    "GraphDistTable",
    "parse_lif",
    "parse_transport_matrix",
    "DeadEnd",
    "GraphTask",
    "GraphAgent",
    "GraphMAPDInstance",
    "GraphMAPD",
    "save_graph_mapd_solution",
    "is_valid_graph_mapd_solution",
    # Backward-compat aliases
    "LifTask",
    "LifAgent",
    "LifMAPDInstance",
    "LifMAPD",
    "save_lif_solution",
    "is_valid_lif_mapd_solution",
]
