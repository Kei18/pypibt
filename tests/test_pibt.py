import numpy as np

from pypibt import PIBT, is_valid_mapf_solution


def test_PIBT():
    grid = np.full((2, 3), True)
    starts = [(0, 0), (0, 2)]
    goals = [(0, 2), (0, 0)]
    pibt = PIBT(grid, starts, goals)
    configs = pibt.run()
    assert is_valid_mapf_solution(grid, starts, goals, configs)
