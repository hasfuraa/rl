#!/usr/bin/env python

from typing import Tuple, List

from abc import ABC, abstractmethod
import torch

class Env(ABC):
    """Environment interface."""

    @abstractmethod
    def _observation(self) -> torch.Tensor:
        """
        Make an observation about the state of the world.

        Returns
        -------
        obs: torch.Tensor
            Observation about the state of the world.
        """
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Takes the agents desired action in the environment.

        Parameters
        ----------
        action: torch.Tensor
            Action requested by the agent.

        Returns
        -------
        obs: torch.Tensor
            Observation about the state of the world.
        reward: float
            Reward for this action.
        is_terminal: bool
            Whether this action caused agent to arrive at terminal point.
        """
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset state of env to starting point.

        Returns
        -------
        obs: torch.Tensor
            Observation of origin state of env.
        """
        pass

    @abstractmethod
    def visualize(self) -> List[List[str]]:
        """
        Return an ASCII visualization of the environment.

        Returns
        -------
        viz: List[List[str]]
            Printable visualization of env.
        """
        pass

    @property
    @abstractmethod
    def n_observations(self) -> int:
        """Return number of observation states."""
        pass

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Return number of action states."""
        pass

class Maze2d(Env):
    LEFT, RIGHT, UP, DOWN = [0, -1], [0, 1], [-1, 0], [1, 0]
    DIRS = [UP, RIGHT, DOWN, LEFT]

    """Environment interface."""
    def __init__(self, maze: List[List[str]]) -> None:
        self.maze = maze
        self.pos = None
        n_rows, n_cols = len(maze), len(maze[0])
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                cell = self.maze[row_idx][col_idx]
                assert cell in (".", "s", "g", "x"), f"Unable to parse maze '{cell}'."
                if self.maze[row_idx][col_idx] == "s":
                    self.start = torch.tensor([row_idx, col_idx])
        self.pos = self.start.clone()
        assert self.pos is not None, "No starting point selected!"

    def _observation(self) -> torch.Tensor:
        """
        One hot observation about the state of the world.

        Returns
        -------
        obs: torch.Tensor
            Observation about the state of the world.
        """
        return self.pos


    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Takes the agents desired action in the environment.

        Parameters
        ----------
        action: torch.Tensor
            Action requested by the agent.

        Returns
        -------
        obs: torch.Tensor
            Observation about the state of the world.
        reward: float
            Reward for this action.
        is_terminal: bool
            Whether this action caused agent to arrive at terminal point.
        """
        reward, is_terminal = 0.0, False
        attempt = self.pos + torch.tensor(self.DIRS[action])

        in_bounds = 1 <= attempt[0] < len(self.maze) and 0 <= attempt[1] < len(self.maze[0])
        if in_bounds and self.maze[attempt[0]][attempt[1]] != "x":
            self.pos = attempt
            reward = 10.0 if self.maze[attempt[0]][attempt[1]] == "g" else 0.0
            is_terminal = self.maze[attempt[0]][attempt[1]] == "g"

        return self.pos, reward, is_terminal
        

    def reset(self) -> torch.Tensor:
        """
        Reset state of env to starting point.

        Returns
        -------
        obs: torch.Tensor
            Observation of origin state of env.
        """
        self.pos = self.start

    def visualize(self) -> List[List[str]]:
        """
        Return an ASCII visualization of the environment.

        Returns
        -------
        viz: List[List[str]]
            Printable visualization of env.
        """
        viz = [x[:] for x in self.maze]
        viz[self.pos[0]][self.pos[1]] = "+"
        return viz

    @property
    def n_observations(self) -> int:
        """Return number of observation states."""
        n_rows, n_cols = len(self.maze), len(self.maze[0])
        return n_rows * n_cols

    @property
    def n_actions(self) -> int:
        """Return number of action states."""
        return len(self.DIRS)

    def grid_to_flat(self, grid: torch.Tensor) -> int:
        n_cols = len(self.maze[0])
        return grid[0] * n_cols + grid[1]

    def flat_to_grid(self, flat: int) -> torch.Tensor:
        n_cols = len(self.maze[0])
        return torch.Tensor([flat // n_cols, flat % n_cols])


def test1() -> None:
    maze_s = [
        ["s", ".", ".", "."],
        [".", "x", ".", "."],
        [".", "x", ".", "."],
        [".", "x", ".", "g"],
    ]
    maze = Maze2d(maze_s)

    # Get to goal.
    maze.reset()
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.RIGHT))
    assert torch.allclose(obs, torch.tensor([0, 1])) 
    assert reward == 0.0 
    assert not is_terminal
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.RIGHT))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.RIGHT))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.DOWN))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.DOWN))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.DOWN))
    assert torch.allclose(obs, torch.tensor([3, 3]))
    assert reward == 10.0 
    assert is_terminal

    # Get blocked.
    maze.reset()
    assert torch.allclose(maze._observation(), torch.tensor([0, 0]))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.DOWN))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.RIGHT))
    assert torch.allclose(maze._observation(), torch.tensor([1, 0]))

    # Out of bounds.
    maze.reset()
    assert torch.allclose(maze._observation(), torch.tensor([0, 0]))
    obs, reward, is_terminal = maze.step(maze.DIRS.index(maze.LEFT))
    assert torch.allclose(maze._observation(), torch.tensor([0, 0]))

    print("Passed test!")

if __name__ == "__main__":
    test1()
