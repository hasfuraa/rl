#!/usr/bin/env python

from typing import List

import torch
from dataclasses import dataclass

from policies import Policy, MLP
from objectives import Objective, Reinforce
from environments import Env, Maze2d


@dataclass
class Traj:
    rewards: List[float]
    log_probs: List[float]
    obs: List[int]

    def append(self, reward: float, log_prob: float, obs: int) -> None:
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.obs.append(obs)


class Trainer(object):
    """RL algorithm to fit model."""
    def __init__(self, policy: Policy, obj: Objective, env: Env, lr: float = 1e-3) -> None:
        self.policy = policy
        self.obj = obj
        self.env = env
        self.opt = torch.optim.Adam(policy.model.parameters(), lr=lr)

    def fit(self, n_eps: int) -> None:
        for ep in range(n_eps):
            self.env.reset()
            obs = self.env._observation()
            is_terminal = False
            traj = Traj(obs=[], rewards=[], log_probs=[])
            while not is_terminal:
                one_hot = torch.nn.functional.one_hot(self.env.grid_to_flat(obs), num_classes=self.env.n_observations)
                action, log_prob = self.policy.get_action(one_hot)
                next_obs, reward, is_terminal = self.env.step(action)
                traj.append(reward=reward, log_prob=log_prob, obs=obs)
                obs = next_obs
            returns = self.obj.compute_returns(torch.tensor(traj.rewards))
            loss = self.obj.compute_loss(returns, torch.stack(traj.log_probs))

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if ep and ep % 10 == 0:
                ep_return = float(sum(traj.rewards))
                print(f"[Ep {ep:4d}] return='{ep_return:6.1f}'.")

    def eval(self) -> None:
        self.env.reset()
        obs = self.env._observation()
        is_terminal = False
        while not is_terminal:
            print("\n".join([str(x) for x in self.env.visualize()]))
            print()
            one_hot = torch.nn.functional.one_hot(self.env.grid_to_flat(obs), num_classes=self.env.n_observations)
            action, log_prob = self.policy.get_action(one_hot)
            next_obs, reward, is_terminal = self.env.step(action)
            obs = next_obs
        print("\n".join([str(x) for x in self.env.visualize()]))

def test1() -> None:
    maze_s = [
        [".", ".", ".", "."],
        [".", "x", "x", "."],
        [".", "x", "x", "."],
        ["s", "x", "x", "g"],
    ]
    env = Maze2d(maze_s)
    model = MLP(n_obs=env.n_observations, n_actions=env.n_actions)
    policy = Policy(model=model)
    obj = Reinforce(gamma=0.1, normalize=False)
    trainer = Trainer(policy=policy, obj=obj, env=env)
    trainer.fit(n_eps=100)
    trainer.eval()
    print("Passed test!")

if __name__ == "__main__":
    test1()
