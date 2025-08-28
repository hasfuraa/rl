#!/usr/bin/env python

from typing import Tuple

from abc import ABC, abstractmethod
import torch

class MLP(torch.nn.Module):
    def __init__(self, n_obs: int, n_actions: int, n_hidden: int = 32) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(*[
            torch.nn.Linear(in_features=n_obs, out_features=n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=n_hidden, out_features=n_actions),
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class Policy(object):
    """Policy interface."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def get_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Use model to predict an action given the observation of the env.

        Parameters
        ----------
        observation: torch.Tensor
            Current obs of the world.

        Parameters
        ----------
        action: int
            Action to be taken in the env.
        log_prob: torch.Tensor
            Log probability of taking this action given the state. Used for loss computation.
        """
        obs_t = obs.unsqueeze(0).float()  # [1, obs_dim]
        logits = self.model(obs_t)  # [1, act_dim]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # [1]
        log_prob = dist.log_prob(action).squeeze(0)  # scalar
        return int(action.item()), log_prob

def test1() -> None:
    model = MLP(n_obs=16, n_actions=4)
    policy = Policy(model=model)
    obs = torch.tensor([0] * 16)
    action, log_prob = policy.get_action(obs)
    print("Passed test!")

if __name__ == "__main__":
    test1()
