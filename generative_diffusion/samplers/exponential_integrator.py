# samplers/exponential_integrator.py
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .base_sampler import BaseSampler
from generative_diffusion.controllable import BaseController


class ExponentialIntegratorSampler(BaseSampler):
    """
    Integrador exponencial (Zhang & Chen, 2023) para SDEs con parte lineal analítica.
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
            t_c = torch.full((x_0.shape[0],), t_curr, device=device, dtype=x_0.dtype)
            t_n = torch.full((x_0.shape[0],), t_next, device=device, dtype=x_0.dtype)

            A = getattr(sde, "exp_matrix", lambda tc, tn: torch.ones_like(tc))(t_c, t_n)
            b = getattr(
                sde,
                "exp_term",
                lambda tc, tn, sm, x: torch.zeros_like(x),
            )(t_c, t_n, score_fn, traj[i])

            cov = getattr(sde, "exp_noise_cov", lambda tc, tn: torch.zeros_like(tc))(
                t_c, t_n
            )
            cov = cov.view(-1, *([1] * (traj[i].ndim - 1)))

            noise = torch.randn_like(traj[i])
            x = (
                A.view(-1, *([1] * (traj[i].ndim - 1))) * traj[i]
                + b
                + torch.sqrt(cov.abs()) * noise
            )

            if controller is not None:
                x = controller.process_step(x_t=x, t=t_n)

            traj[i + 1] = x

        return times, traj
