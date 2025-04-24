# NO FUNCIONA

# schedulers/cosine_scheduler.py
from __future__ import annotations
import torch
from torch import Tensor
from .base_scheduler import BaseScheduler


class CosineScheduler(BaseScheduler):
    """
    Scheduler de ruido con "cosine schedule" ajustado para `t` normalizado en [0, 1].
    Hiperparámetros alineados con el paper original (Nichol & Dhariwal, 2021):

      ᾱ(t) = cos²( π/2 · (t/T + s)/(1 + s) ) / cos²( π/2 · s/(1 + s) )
      β(t) = (π / (1 + s)) · tan( π/2 · (t/T + s)/(1 + s) )

    Se añade clamp para evitar inestabilidades numéricas cerca de t = T.
    """

    def __init__(self, *, T: int = 1000, s: float = 0.008) -> None:
        super().__init__(T=T)
        self.s = float(s)
        self._denom_tensor = None

    def alpha_bar(self, t: Tensor) -> Tensor:
        pi = torch.tensor(torch.pi, device=t.device, dtype=t.dtype)
        scaled_t = (t + self.s) / (1 + self.s)
        u = (pi / 2) * scaled_t
        alpha = torch.cos(u) ** 2

        # Cálculo del denominador (solo una vez)
        if self._denom_tensor is None or self._denom_tensor.device != t.device:
            u0 = (pi / 2) * (self.s / (1 + self.s))
            denom_tensor = torch.cos(u0) ** 2
            self._denom_tensor = denom_tensor.to(device=t.device, dtype=t.dtype)

        return alpha / self._denom_tensor

    def beta(self, t: Tensor) -> Tensor:
        pi = torch.tensor(torch.pi, device=t.device, dtype=t.dtype)
        scaled_t = torch.clamp(((t + self.s) / (1 + self.s)), 0, 0.999)
        u = (pi / 2) * scaled_t

        coef = pi / (1 + self.s)
        return (coef / self.T) * torch.tan(u)
