from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .mapf_utils import Coord, Grid, get_neighbors, is_valid_coord


@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    Q: deque = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix

    def __post_init__(self):
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.grid.size, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if self.table[target] < self.table.size:
            return self.table[target]

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.grid.size
