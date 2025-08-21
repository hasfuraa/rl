#!/usr/bin/env python

from typing import Tuple

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

    @abstractmethod
    @property
    def n_observations(self) -> int:
        """Return number of observation states."""
        pass

    @abstractmethod
    @property
    def n_actions(self) -> int:
        """Return number of action states."""
        pass
