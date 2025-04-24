# sde/subvp_sde.py
from __future__ import annotations

import torch
from torch import Tensor

from .scheduler_sde import SchedulerBasedSDE
from generative_diffusion.schedulers import BaseScheduler


class SubVPSDE(SchedulerBasedSDE):
    """
    Sub‑Variance‑Preserving SDE (γ ∈ (0,1); Song et al., 2021).
    """

    def __init__(self, scheduler: BaseScheduler) -> None:
        super().__init__(scheduler)

    # ------------------------------------------------------------------ #
    def beta_t(self, t: Tensor) -> Tensor:
        return self.scheduler.beta(t)

    # Forward
    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        beta = self._broadcast(self.beta_t(t), x_t)
        return -0.5 * beta * x_t

    def diffusion(self, t: Tensor) -> Tensor:
        alpha_bar = self.scheduler.alpha_bar(t)
        return torch.sqrt(self.beta_t(t) * (1.0 - alpha_bar**4))

    def mu_t(self, x_0: Tensor, t: Tensor) -> Tensor:
        a_bar = self._broadcast(self.scheduler.alpha_bar(t), x_0)
        return torch.sqrt(a_bar) * x_0

    def sigma_t(self, t: Tensor) -> Tensor:
        a_bar = self.scheduler.alpha_bar(t)
        return torch.sqrt((1.0 - a_bar) * (1.0 - a_bar**4))

    # Backward
    def backward_drift(self, x_t: Tensor, t: Tensor, score_fn) -> Tensor:
        beta = self._broadcast(self.beta_t(t), x_t)
        g = self._broadcast(self.diffusion(t), x_t)
        return -0.5 * beta * x_t - (g**2) * score_fn(x_t, t)
