# NO FUNCIONA

# samplers/exponential_integrator.py
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .base_sampler import BaseSampler
from generative_diffusion.controllable import BaseController


class ExponentialIntegratorSampler(BaseSampler):
    """
    Integrador exponencial (Zhang & Chen, 2023) para SDEs con parte lineal analítica.
    """

    def sample(
        self,
        x_0: Tensor,
        sde,
        score_model: Callable,
        *,
        t_0: float = 1.0,
        t_end: float = 1e-3,
        n_steps: int = 500,
        condition: Optional[Tensor] = None,
        seed: Optional[int] = None,
        controller: Optional[BaseController] = None,
    ) -> Tuple[Tensor, Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        device = x_0.device
        times = torch.linspace(t_0, t_end, n_steps + 1, device=device)

        traj = torch.empty(n_steps + 1, *x_0.shape, device=device, dtype=x_0.dtype)
        traj[0] = x_0

        score_fn = self._prepare_score_model(score_model, condition)

        for i in range(n_steps):
            t_curr, t_next = times[i], times[i + 1]
            # Normalizar tiempos si la SDE lo requiere (ej: t ∈ [0, 1])
            t_c_normalized = t_curr / sde.T if hasattr(sde, "T") else t_curr
            t_n_normalized = t_next / sde.T if hasattr(sde, "T") else t_next

            t_c = torch.full(
                (x_0.shape[0],), t_c_normalized, device=device, dtype=x_0.dtype
            )
            t_n = torch.full(
                (x_0.shape[0],), t_n_normalized, device=device, dtype=x_0.dtype
            )

            # Verificar existencia de métodos en la SDE
            if not hasattr(sde, "exp_matrix"):
                raise AttributeError("La SDE debe implementar 'exp_matrix'.")
            if not hasattr(sde, "exp_term"):
                raise AttributeError("La SDE debe implementar 'exp_term'.")
            if not hasattr(sde, "exp_noise_cov"):
                raise AttributeError("La SDE debe implementar 'exp_noise_cov'.")

            A = sde.exp_matrix(t_c, t_n)  # Factor de decaimiento exponencial
            b = sde.exp_term(
                t_c, t_n, score_fn, traj[i]
            )  # Término no lineal (score-driven)
            cov = sde.exp_noise_cov(t_c, t_n)  # Covarianza del ruido

            # Asegurar que la covarianza no sea negativa
            cov = cov.clamp(min=0.0).view(-1, *([1] * (traj[i].ndim - 1)))

            noise = torch.randn_like(traj[i])
            x = (
                A.view(-1, *([1] * (traj[i].ndim - 1))) * traj[i]
                + b
                + torch.sqrt(cov) * noise  # Eliminar .abs() gracias al clamp
            )

            if controller is not None:
                x = controller.process_step(x_t=x, t=t_n)

            traj[i + 1] = x

        return times, traj
