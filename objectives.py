#!/usr/bin/env python

from typing import List
from abc import ABC, abstractmethod

import torch

class Objective(ABC):
    @abstractmethod
    def compute_returns(rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute returns given per step rewards.
        """
        raise NotImplementedError("Not Implemented!")
    
    @abstractmethod
    def compute_loss(returns: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute loss given returns and log_probs.
        """
        raise NotImplementedError("Not Implemented!")


class Reinforce(Objective):
    def __init__(self, gamma: float, normalize: bool) -> None:
        self.gamma = gamma
        self.normalize = normalize

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute returns given per step rewards.
        """
        G = 0.0
        out: List[float] = []
        for r in reversed(rewards):
            G = float(r) + self.gamma * G
            out.append(G)
        out.reverse()
        returns = torch.tensor(out, dtype=torch.float32)
        if self.normalize and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        return returns
    
    def compute_loss(self, returns: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute loss given returns and log_probs.
        """
        return -torch.sum(returns * log_probs)

def test1() -> None:
    reinforce = Reinforce(gamma=0.1, normalize=False)
    rewards = [0.0] * 10
    rewards[-1] = 10.0
    returns = reinforce.compute_returns(torch.tensor(rewards))
    assert returns[-1] == 10.0
    assert returns[-2] == 1.0
    assert returns[-3] == 0.1
    assert reinforce.compute_loss(returns, torch.ones_like(returns))
    print("Passed test!")

if __name__ == "__main__":
    test1()
