# schedulers/cosine_scheduler.py
from __future__ import annotations

import torch
from torch import Tensor

from .base_scheduler import BaseScheduler


class CosineScheduler(BaseScheduler):
    """
    Scheduler de ruido con "cosine schedule" ajustado para `t` normalizado en [0, 1].
    Hiperparámetros alineados con el paper original (Nichol & Dhariwal, 2021).

    Usando t normalizado:
      ᾱ(t) = cos²( π/2 · (t + s)/(1 + s) ) / cos²( π/2 · s/(1 + s) )
      β_norm(t) = (π / (1 + s)) · tan( π/2 · (t + s)/(1 + s) )

    Se añade clamp para evitar inestabilidades numéricas cerca de t = 1.
    """

    def __init__(self, *, s: float = 0.008) -> None:
        self.s = float(s)
        self._denom_tensor = None
        self._pi = torch.tensor(torch.pi)
        self._coef = self._pi / (1 + self.s)

    def _get_pi(self, t: Tensor) -> Tensor:
        """Devuelve pi en el dispositivo y dtype correctos."""
        return self._pi.to(device=t.device, dtype=t.dtype)

    def _get_coef(self, t: Tensor) -> Tensor:
        """Devuelve el coeficiente pi/(1+s) en el dispositivo y dtype correctos."""
        return self._coef.to(device=t.device, dtype=t.dtype)

    def alpha_bar(self, t: Tensor) -> Tensor:
        pi = self._get_pi(t)
        scaled_t = (t + self.s) / (1 + self.s)
        f_t = torch.cos((pi / 2) * scaled_t) ** 2

        # Cálculo del denominador f(0) (solo una vez por dispositivo/dtype)
        # Se usa un tensor persistente para evitar recalcularlo
        if (
            self._denom_tensor is None
            or self._denom_tensor.device != t.device
            or self._denom_tensor.dtype != t.dtype
        ):
            f_0 = torch.cos((pi / 2) * (self.s / (1 + self.s))) ** 2
            self._denom_tensor = f_0.to(device=t.device, dtype=t.dtype)

        # Evitar división por cero si f(0) es muy pequeño
        # (aunque con s=0.008 no debería pasar)
        alpha = f_t / torch.max(
            self._denom_tensor, torch.tensor(self._EPS, device=t.device, dtype=t.dtype)
        )
        # Clamp final para asegurar que esté en [0, 1]
        # debido a posibles errores numéricos
        return torch.clamp(alpha, 0.0, 1.0)

    def beta(self, t: Tensor) -> Tensor:
        # Usar la fórmula derivada para beta_norm(t) con t normalizado
        pi = self._get_pi(t)
        coef = self._get_coef(t)

        # Aplicar clamp interno a scaled_t para evitar tan(pi/2)
        # Clamp a un valor ligeramente menor que 1.0
        scaled_t = torch.clamp((t + self.s) / (1 + self.s), min=0.0, max=0.9999)
        u = (pi / 2) * scaled_t

        # beta_norm = (pi / (1 + s)) * tan(u)
        beta_t = coef * torch.tan(u)

        # β(t) debe estar acotado para evitar inestabilidades, siguiendo DDPM/DDIM.
        # El paper original de cosine schedule deriva beta de alpha_bar discretizado:
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, acotado a <= 0.999
        # Aquí usamos la fórmula continua derivada, pero la acotamos igualmente.
        # El valor máximo de 0.999 es una heurística común.
        return torch.clamp(beta_t, min=self._EPS, max=0.999)  # Acotar beta
