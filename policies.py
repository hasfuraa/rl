#!/usr/bin/env python

from typing import Tuple

from abc import ABC, abstractmethod
import torch

class Policy(ABC):
    """Policy interface."""

    @abstractmethod
    def get_action(observation: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Use policy model to predict an action given the observation of the env.

        Parameters
        ----------
        observation: torch.Tensor
            Current obs of the world.

        Parameters
        ----------
        action: int
            Action to be taken in the env.
        logits: torch.Tensor
            Logit values for each of the possible actions.
        """
        pass

    def udpate(loss: torch.Tensor) -> None:
        pass
