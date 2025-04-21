# schedulers/linear_scheduler.py
from __future__ import annotations

import torch
from torch import Tensor

from .base_scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    """
    Scheduler lineal: β(t) = β_min + (β_max − β_min) · t.
    """

    def __init__(
        self,
        *,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        T: int = 1000,
    ) -> None:
        if not (0.0 < beta_min < beta_max < 1.0):
            raise ValueError("Se requiere 0 < beta_min < beta_max < 1.")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        super().__init__(T=T)

    # ------------------------------------------------------------------ #
    # Implementación de la interfaz                                      #
    # ------------------------------------------------------------------ #
    def beta(self, t: Tensor) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t: Tensor) -> Tensor:
        integrated_beta = (
            self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        )
        return torch.exp(-integrated_beta)
