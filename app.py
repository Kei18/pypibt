import argparse
import os

from pypibt import (
    PIBT,
    get_grid,
    get_scenario,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "random-32-32-10.map"
        ),
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "assets", "random-32-32-10-random-1.scen"
        ),
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=1000)
    args = parser.parse_args()

    # define problem instance
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    # solve MAPF
    pibt = PIBT(grid, starts, goals, seed=args.seed)
    plan = pibt.run(max_timestep=args.max_timestep)

    # validation: True -> feasible solution
    print(f"solved: {is_valid_mapf_solution(grid, starts, goals, plan)}")

    # save result
    save_configs_for_visualizer(plan, args.output_file)
