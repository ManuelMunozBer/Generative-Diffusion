# schedulers/cosine_scheduler.py
from __future__ import annotations

import math

import torch
from torch import Tensor

from .base_scheduler import BaseScheduler


class CosineScheduler(BaseScheduler):
    """
    Scheduler coseno de Nichol & Dhariwal (2021):

    ᾱ(t) = cos²[ (π/2)·(t + s)/(1 + s) ]   ― con s ≈ 0.008.
    """

    def __init__(self, *, s: float = 0.008, T: int = 1000) -> None:
        if s < 0.0:
            raise ValueError("`s` debe ser no negativo.")
        self.s = float(s)
        super().__init__(T=T)

    # ------------------------------------------------------------------ #
    # Implementación de la interfaz                                      #
    # ------------------------------------------------------------------ #
    def alpha_bar(self, t: Tensor) -> Tensor:
        arg = (math.pi / 2.0) * ((t + self.s) / (1.0 + self.s))
        return torch.cos(arg) ** 2

    def beta(self, t: Tensor) -> Tensor:
        """
        β(t) = -d/dt ln ᾱ(t) = (π / (1 + s)) · tan(arg)
        (con t normalizado en [0,1]).
        """
        arg = (math.pi / 2.0) * ((t + self.s) / (1.0 + self.s))
        beta_cont = (math.pi / (1.0 + self.s)) * torch.tan(arg)

        # Limitar para estabilidad numérica
        limit = torch.tensor(0.999, device=t.device, dtype=t.dtype)
        return torch.minimum(beta_cont, limit)
