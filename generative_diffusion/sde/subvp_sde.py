# sde/subvp_sde.py
from __future__ import annotations

import torch
from torch import Tensor

from .scheduler_sde import SchedulerBasedSDE
from generative_diffusion.schedulers import BaseScheduler


class SubVPSDE(SchedulerBasedSDE):
    """
    Sub‑Variance‑Preserving SDE (γ ∈ (0,1); Song et al., 2021).
    """

    def __init__(self, scheduler: BaseScheduler, *, gamma: float = 0.5) -> None:
        if not (0.0 < gamma < 1.0):
            raise ValueError("`gamma` debe estar en (0, 1).")
        super().__init__(scheduler)
        self.gamma = float(gamma)

    # ------------------------------------------------------------------ #
    def beta_t(self, t: Tensor) -> Tensor:
        return self.scheduler.beta(t)

    # Forward
    def drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        beta = self.beta_t(t).view(-1, *([1] * (x_t.ndim - 1)))
        return -0.5 * beta * x_t

    def diffusion(self, t: Tensor) -> Tensor:
        return torch.sqrt(self.beta_t(t) * (1.0 - self.gamma))

    def mu_t(self, x_0: Tensor, t: Tensor) -> Tensor:
        a_bar = self.scheduler.alpha_bar(t).view(-1, *([1] * (x_0.ndim - 1)))
        return torch.sqrt(a_bar) * x_0

    def sigma_t(self, t: Tensor) -> Tensor:
        a_bar = self.scheduler.alpha_bar(t)
        return torch.sqrt((1.0 - a_bar) * (1.0 - self.gamma))

    # Backward
    def backward_drift(self, x_t: Tensor, t: Tensor, score_fn) -> Tensor:
        beta = self.beta_t(t).view(-1, *([1] * (x_t.ndim - 1)))
        return -0.5 * beta * (x_t + (1.0 - self.gamma) * score_fn(x_t, t))
