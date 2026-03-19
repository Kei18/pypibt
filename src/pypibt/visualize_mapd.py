#!/usr/bin/env python3
"""MAPD Visualization Tool using Matplotlib.

This module provides visualization for Multi-Agent Pickup and Delivery (MAPD)
solutions, showing agent trajectories, task assignments, pickup/delivery locations,
and task completion status.

Usage:
    # Static visualization
    python scripts/visualize_mapd.py output_mapd.txt -m assets/maps/random-32-32-10.map

    # Interactive animation
    python scripts/visualize_mapd.py output_mapd.txt -m assets/maps/random-32-32-10.map --animate

    # Save as GIF
    python scripts/visualize_mapd.py output_mapd.txt -m assets/maps/random-32-32-10.map --save demo.gif
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pypibt import get_grid


def load_solution(file_path: str) -> tuple[list[list[tuple[int, int]]], list[list[tuple[tuple[int, int], str]]]]:
    """Load solution from file with agent goals and status.
    
    Args:
        file_path: Path to solution file.
        
    Returns:
        Tuple of (solution, agent_goals) where:
        - solution: List of configurations, each a list of (x,y) positions
        - agent_goals: List of per-timestep goals: [[(goal, status), ...], ...]
          status is 'A' (assigned/carrying), 'T' (targeting), 'F' (free), or None if not available
    """
    solution = []
    agent_goals = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse: timestep:(x,y)[gx,gy,status],(x,y)[gx,gy,status],...
        if ':' not in line:
            continue
        
        parts = line.split(':')
        if len(parts) != 2:
            continue
        
        config_str = parts[1].strip().rstrip(',')
        positions = []
        goals = []
        
        # Parse positions and optional goals
        import re
        # Match (x,y) optionally followed by [gx,gy,status]
        pattern = r'\((\d+),(\d+)\)(?:\[(\d+),(\d+),([ATF])\])?'
        for match in re.finditer(pattern, config_str):
            x, y = int(match.group(1)), int(match.group(2))
            positions.append((x, y))
            
            if match.group(3) and match.group(4) and match.group(5):
                gx, gy = int(match.group(3)), int(match.group(4))
                status = match.group(5)
                goals.append(((gx, gy), status))
            else:
                goals.append((None, None))
        
        solution.append(positions)
        agent_goals.append(goals)
    
    return solution, agent_goals


def load_tasks_from_mapd(mapd_instance):
    """Extract task information from MAPD instance.
    
    Args:
        mapd_instance: MAPD solver instance with completed_tasks.
    
    Returns:
        List of task dictionaries with pickup, delivery, and timing info.
    """
    tasks = []
    for agent_id, task in mapd_instance.completed_tasks:
        tasks.append({
            'agent_id': agent_id,
            'task_id': task.id,
            'pickup': (task.loc_pickup[1], task.loc_pickup[0]),  # (x, y)
            'delivery': (task.loc_delivery[1], task.loc_delivery[0]),  # (x, y)
            'appear': task.timestep_appear,
            'finish': task.timestep_finished
        })
    return tasks


class MAPDVisualizer:
    """Interactive MAPF/MAPD solution visualizer using matplotlib."""
    
    def __init__(self, grid: np.ndarray, solution: list[list[tuple[int, int]]], 
                 agent_goals: list[list[tuple[tuple[int, int], str]]] = None,
                 interpolation_steps: int = 1,
                 mode: str = 'auto'):
        """Initialize visualizer.
        
        Args:
            grid: Grid with obstacles (True = free, False = obstacle).
            solution: List of agent positions over time.
            agent_goals: Optional list of per-timestep goals and status: [[(goal, status), ...], ...]
            interpolation_steps: Number of interpolated frames per timestep (1 = no interpolation).
            mode: Visualization mode - 'mapd', 'mapf', or 'auto' (default: 'auto' detects from goals).
        """
        self.grid = grid
        self.solution = solution
        self.agent_goals = agent_goals or []
        self.n_agents = len(solution[0]) if solution else 0
        self.n_timesteps = len(solution)
        self.colors = plt.cm.tab20(np.linspace(0, 1, self.n_agents))
        self.interpolation_steps = max(1, interpolation_steps)
        
        # Determine visualization mode
        if mode.lower() == 'mapd':
            self.is_mapd = True
        elif mode.lower() == 'mapf':
            self.is_mapd = False
        else:  # auto mode
            # Determine if MAPD based on presence of agent goals
            self.is_mapd = len(self.agent_goals) > 0 and any(
                any(g[0] is not None for g in timestep_goals) 
                for timestep_goals in self.agent_goals
            )
        
        # Count completed tasks for both MAPF and MAPD
        self.tasks_completed, self.total_tasks = self._count_completed_tasks()
    
    def _count_completed_tasks(self) -> tuple[list[int], int]:
        """Count cumulative completed tasks at each timestep and estimate total tasks.
        
        For MAPD: A task is completed when an agent transitions from
        'A' (assigned/carrying) to 'F' (free) or 'T' (new task).
        
        For MAPF: A task is completed when an agent reaches its goal.
        
        Returns:
            Tuple of (completed_counts, total_tasks) where:
            - completed_counts: List of cumulative task counts at each timestep
            - total_tasks: Estimated total number of unique tasks
        """
        completed_counts = [0]  # Start with 0 tasks completed
        total_completed = 0
        
        if self.is_mapd:
            # Track unique task assignments to estimate total
            task_assignments = set()  # Set of (agent_id, goal_tuple)
            
            for t in range(len(self.agent_goals)):
                curr_goals = self.agent_goals[t]
                
                # Track task assignments
                for i, (goal, status) in enumerate(curr_goals):
                    if goal and status in ('A', 'T'):  # Agent has a task
                        task_assignments.add((i, tuple(goal)))
                
                # Count completions
                if t > 0:
                    prev_goals = self.agent_goals[t-1]
                    
                    for i in range(min(len(prev_goals), len(curr_goals))):
                        prev_goal, prev_status = prev_goals[i]
                        curr_goal, curr_status = curr_goals[i]
                        
                        # Task completed when:
                        # 1. Was carrying (A) and now free (F)
                        # 2. Was carrying (A) and now targeting new task (T) with different goal
                        if prev_status == 'A' and curr_status == 'F':
                            total_completed += 1
                        elif prev_status == 'A' and curr_status in ('T', 'A') and prev_goal != curr_goal:
                            # Delivered and immediately got new task
                            total_completed += 1
                
                completed_counts.append(total_completed)
            
            # Estimate total tasks as max of completed + currently assigned
            total_tasks = total_completed + sum(
                1 for _, status in self.agent_goals[-1] if status in ('A', 'T')
            )
        else:
            # MAPF: Count agents reaching their final goals
            if len(self.solution) == 0:
                return [0], 0
            
            # Get final goal positions for all agents
            final_goals = self.solution[-1]
            agents_at_goal = [False] * self.n_agents
            
            for t in range(len(self.solution)):
                curr_positions = self.solution[t]
                
                # Check each agent if they reached their goal
                for i, pos in enumerate(curr_positions):
                    if not agents_at_goal[i]:
                        # Agent hasn't completed yet, check if at goal
                        if pos == final_goals[i]:
                            agents_at_goal[i] = True
                            total_completed += 1
                
                completed_counts.append(total_completed)
            
            # For MAPF, total tasks = number of agents (each has one goal)
            total_tasks = self.n_agents
        
        return completed_counts, total_tasks
        
    def _draw_grid(self, ax: plt.Axes) -> None:
        """Draw the map grid with obstacles."""
        height, width = self.grid.shape
        
        # Draw obstacles
        for y in range(height):
            for x in range(width):
                if not self.grid[y, x]:
                    ax.add_patch(Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor='gray', edgecolor='black', linewidth=0.5
                    ))
        
        # Grid lines
        for x in range(width + 1):
            ax.plot([x - 0.5, x - 0.5], [-0.5, height - 0.5], 
                        'k-', linewidth=0.3, alpha=0.3)
        for y in range(height + 1):
            ax.plot([-0.5, width - 0.5], [y - 0.5, y - 0.5],
                        'k-', linewidth=0.3, alpha=0.3)
        
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match visualization convention
        
    def _draw_trajectories(self, ax: plt.Axes, alpha=0.3) -> None:
        """Draw complete agent trajectories as lines."""
        for agent_id in range(self.n_agents):
            path = [(self.solution[t][agent_id][0], self.solution[t][agent_id][1]) 
                   for t in range(self.n_timesteps)]
            path = np.array(path)
            
            ax.plot(path[:, 0], path[:, 1], 
                        color=self.colors[agent_id],
                        alpha=alpha, linewidth=1.5, 
                        label=f'Agent {agent_id}' if agent_id < 10 else '')
    
    def _draw_tasks(self, ax: plt.Axes, t: int) -> None:
        """Draw pickup and delivery locations for MAPD (currently disabled).
        
        Args:
            ax: Matplotlib axes.
            t: Current timestep.
        """
        # Task locations are no longer visualized - only goal lines are shown
        pass
    
    def _draw_current_state(self, ax: plt.Axes, positions: list[tuple[float, float]], goal_timestep: int) -> list:
        """Draw agents at current (possibly interpolated) positions.
        
        Args:
            ax: Matplotlib axes.
            positions: List of agent positions (x, y) - can be interpolated floats.
            goal_timestep: Timestep to use for determining agent status.
            
        Returns:
            List of artist objects for blitting.
        """
        artists = []
        
        for i, (x, y) in enumerate(positions):
            color = self.colors[i]
            
            # Get agent status from goals if available
            status = None
            goal = None
            at_goal = False
            
            if self.is_mapd:
                if goal_timestep < len(self.agent_goals) and i < len(self.agent_goals[goal_timestep]):
                    goal, status = self.agent_goals[goal_timestep][i]
                    if goal:
                        at_goal = (abs(x - goal[0]) + abs(y - goal[1])) < 0.01
            else:
                # MAPF: Check if at final goal
                if len(self.solution) > 0 and i < len(self.solution[-1]):
                    goal = self.solution[-1][i]
                    at_goal = (abs(x - goal[0]) + abs(y - goal[1])) < 0.01
            
            # Determine circle style
            if self.is_mapd:
                if status == 'A':  # Assigned/carrying
                    # Filled circle when carrying item
                    artist, = ax.plot(x, y, 'o', color=color, markersize=8, 
                           markeredgecolor=color, markeredgewidth=1.5)
                elif status == 'T':  # Targeting pickup
                    # Unfilled circle when moving to pickup
                    artist, = ax.plot(x, y, 'o', color='white', markersize=8,
                           markeredgecolor=color, markeredgewidth=1.5)
                elif status == 'F':  # Free (idle)
                    # Hollow circle without filling - lowest priority agents
                    artist, = ax.plot(x, y, 'o', color='white', markersize=8,
                           markeredgecolor=color, markeredgewidth=1.5, alpha=0.6)
                else:
                    # Fallback
                    artist, = ax.plot(x, y, 'o', color=color, markersize=8,
                           markeredgecolor=color, markeredgewidth=1.5)
            else:
                # MAPF: Filled while moving, hollow when at goal
                if at_goal:
                    # Hollow circle at goal
                    artist, = ax.plot(x, y, 'o', color='white', markersize=8,
                           markeredgecolor=color, markeredgewidth=1.5)
                else:
                    # Filled circle while moving to goal
                    artist, = ax.plot(x, y, 'o', color=color, markersize=8,
                           markeredgecolor=color, markeredgewidth=1.5)
            
            artists.append(artist)
        
        return artists
    
    def _draw_goal_lines(self, ax: plt.Axes, positions: list[tuple[float, float]], goal_timestep: int) -> list:
        """Draw straight lines from agents to their goals.
        
        For MAPD: Shows current dynamic goals (pickup or delivery locations).
                  Free agents have no goal lines as they're idle.
        For MAPF: Shows final static goals from the last timestep.
        
        Args:
            ax: Matplotlib axes.
            positions: List of agent positions (x, y) - can be interpolated floats.
            goal_timestep: Timestep to use for goal locations.
            
        Returns:
            List of artist objects for blitting.
        """
        artists = []
        
        for i, (x, y) in enumerate(positions):
            color = self.colors[i]
            goal = None
            status = None
            
            if self.is_mapd:
                # MAPD: Get current dynamic goal and status from agent_goals
                if goal_timestep < len(self.agent_goals) and i < len(self.agent_goals[goal_timestep]):
                    goal, status = self.agent_goals[goal_timestep][i]
                    # Free agents should not show goal lines (they're idle)
                    if status == 'F':
                        continue
            else:
                # MAPF: Always show goal lines to final positions
                if len(self.solution) > 0 and i < len(self.solution[-1]):
                    goal = self.solution[-1][i]
            
            # Draw line to goal if goal exists and agent is not at goal
            if goal is not None:
                # Check if agent is at goal (allow small tolerance for interpolation)
                dist = abs(x - goal[0]) + abs(y - goal[1])
                if dist > 0.01:  # Only draw if not at goal (small tolerance for floating point)
                    line, = ax.plot([x, goal[0]], [y, goal[1]], '-', color=color, 
                           linewidth=1.0, alpha=0.7)
                    artists.append(line)
        
        return artists
    
    def _get_interpolated_state(self, timestep: int, alpha: float) -> tuple[list[tuple[float, float]], int]:
        """Get interpolated agent positions between timesteps.
        
        Args:
            timestep: Base timestep index.
            alpha: Interpolation factor [0, 1] where 0=start of timestep, 1=end.
            
        Returns:
            Tuple of (interpolated_positions, goal_timestep) where goal_timestep is the 
            timestep to use for goal visualization.
        """
        if timestep >= self.n_timesteps - 1 or alpha >= 1.0:
            return self.solution[timestep], timestep
        
        # Get positions at current and next timestep
        pos_current = self.solution[timestep]
        pos_next = self.solution[timestep + 1]
        
        # Interpolate each agent's position
        interpolated = []
        for (x1, y1), (x2, y2) in zip(pos_current, pos_next):
            x_interp = x1 + (x2 - x1) * alpha
            y_interp = y1 + (y2 - y1) * alpha
            interpolated.append((x_interp, y_interp))
        
        # Use next timestep's goals when alpha > 0.5, otherwise current timestep
        goal_timestep = timestep + 1 if alpha > 0.5 else timestep
        
        return interpolated, goal_timestep
    
    def plot_static(self, show_trajectories=True):
        """Create static visualization showing final state.
        
        Args:
            show_trajectories: Whether to show agent paths.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        self._draw_grid(ax)
        
        # Show final positions
        final_t = self.n_timesteps - 1
        final_positions = self.solution[final_t]
        self._draw_current_state(ax, final_positions, final_t)
        self._draw_goal_lines(ax, final_positions, final_t)
        
        mode_str = "MAPD" if self.is_mapd else "MAPF"
        ax.set_title(f'{mode_str} Solution - Final State (T={final_t})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.tight_layout()
        return fig
    
    def animate(self, interval=200, save_path=None):
        """Create interactive animation with interpolation and blitting for performance.
        
        Args:
            interval: Milliseconds between frames.
            save_path: Optional path to save animation (e.g., 'demo.gif').
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw static elements once (grid)
        self._draw_grid(ax)
        
        # Initialize title with task completion info
        mode_str = "MAPD" if self.is_mapd else "MAPF"
        if self.tasks_completed and self.total_tasks > 0:
            title_text = f'{mode_str} - Timestep: 0/{self.n_timesteps - 1} | Tasks: 0/{self.total_tasks}'
        else:
            title_text = f'{mode_str} - Timestep: 0/{self.n_timesteps - 1}'
        title = ax.set_title(title_text, fontsize=12, fontweight='bold')
        
        # Calculate total number of frames
        total_frames = (self.n_timesteps - 1) * self.interpolation_steps + 1
        
        # Store artists that will be updated
        dynamic_artists = []
        
        def init():
            """Initialize animation."""
            return []
        
        def update(frame):
            """Update frame with blitting for performance."""
            nonlocal dynamic_artists
            
            # Remove previous dynamic artists
            for artist in dynamic_artists:
                artist.remove()
            dynamic_artists = []
            
            # Convert frame index to timestep and alpha
            timestep = frame // self.interpolation_steps
            alpha = (frame % self.interpolation_steps) / self.interpolation_steps
            
            # Get interpolated state
            positions, goal_timestep = self._get_interpolated_state(timestep, alpha)
            
            # Draw dynamic elements (agents and goal lines)
            agent_artists = self._draw_current_state(ax, positions, goal_timestep)
            line_artists = self._draw_goal_lines(ax, positions, goal_timestep)
            
            dynamic_artists = agent_artists + line_artists
            
            # Update title with timestep and task completion
            if self.tasks_completed and self.total_tasks > 0:
                # Use timestep (not interpolated) for task count
                tasks_done = self.tasks_completed[min(timestep, len(self.tasks_completed) - 1)]
                title.set_text(f'{mode_str} - Timestep: {timestep}/{self.n_timesteps - 1} | Tasks: {tasks_done}/{self.total_tasks}')
            else:
                title.set_text(f'{mode_str} - Timestep: {timestep}/{self.n_timesteps - 1}')
            
            # Return all artists including title for blitting
            return dynamic_artists + [title]
        
        # Use blit=False to ensure title updates properly
        # cache_frame_data=False helps with real-time performance
        anim = animation.FuncAnimation(
            fig, update, frames=total_frames,
            interval=interval, repeat=True, init_func=init, blit=False,
            cache_frame_data=False
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            else:
                anim.save(save_path, fps=1000//interval)
            print("Done!")
        
        return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description='Visualize MAPF/MAPD solutions with matplotlib',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('input_file', help='Path to solution file (output_mapf.txt or output_mapd.txt)')
    parser.add_argument('-m', '--map', required=True, help='Path to map file')
    parser.add_argument('--mode', choices=['mapf', 'mapd', 'auto'], default='auto',
                        help='Visualization mode: mapf, mapd, or auto-detect (default: auto)')
    parser.add_argument('--save', help='Save animation to file (e.g., demo.gif or demo.mp4)')
    parser.add_argument('--speed', type=float, default=2.0, 
                        help='Animation speed in timesteps per second for saved files (default: 2.0, only used with --save)')
    parser.add_argument('--no-interpolation', action='store_true',
                        help='Disable smooth interpolation for faster real-time playback (5x faster)')
    
    args = parser.parse_args()
    
    # Interpolation: 5 frames per timestep for smooth animation, or 1 for fast playback
    interpolation_steps = 1 if args.no_interpolation else 5
    
    # Calculate interval in milliseconds from timesteps per second
    # For interactive display, use a fixed comfortable interval (20ms = 50 fps for smooth display)
    # For saving, use the user-specified speed
    if args.save:
        # interval = (1000 ms/sec) / (timesteps/sec * frames/timestep)
        interval = int(1000 / (args.speed * interpolation_steps))
    else:
        # Fixed interval for interactive display (responsive GUI)
        interval = 20
    
    # Load map
    print(f"Loading map from {args.map}...")
    grid = get_grid(args.map)
    
    # Load solution
    print(f"Loading solution from {args.input_file}...")
    solution, agent_goals = load_solution(args.input_file)
    
    # Display mode information
    if args.mode == 'auto':
        is_mapd = len(agent_goals) > 0 and any(
            any(g[0] is not None for g in timestep_goals) 
            for timestep_goals in agent_goals
        )
        mode_str = "MAPD" if is_mapd else "MAPF"
        print(f"Auto-detected mode: {mode_str}")
    else:
        mode_str = args.mode.upper()
        print(f"Using mode: {mode_str}")
    
    print(f"Timesteps: {len(solution)}")
    print(f"Agents: {len(solution[0]) if solution else 0}")
    
    # Create visualizer with fixed interpolation
    visualizer = MAPDVisualizer(grid, solution, agent_goals=agent_goals, 
                                 interpolation_steps=interpolation_steps,
                                 mode=args.mode)
    
    # Display task information for MAPF and MAPD
    if visualizer.total_tasks > 0:
        print(f"Total tasks: {visualizer.total_tasks}")
    
    if args.save:
        print(f"Saving with speed: {args.speed} timesteps/second (interval: {interval}ms per frame)")
    else:
        print(f"Interactive display mode (speed parameter ignored, use --save to control speed)")
    
    # Always animate
    fig, anim = visualizer.animate(interval=interval, save_path=args.save)
    if not args.save:
        plt.show()


if __name__ == '__main__':
    main()
