# samplers/probability_flow_ode.py
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .base_sampler import BaseSampler
from generative_diffusion.controllable import BaseController


class ProbabilityFlowODESampler(BaseSampler):
    """
    ODE determinista equivalente a la SDE (Song et al., 2021).
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
        dt = times[1] - times[0]

        traj = torch.empty(n_steps + 1, *x_0.shape, device=device, dtype=x_0.dtype)
        traj[0] = x_0

        score_fn = self._prepare_score_model(score_model, condition)

        for i, t in enumerate(times[:-1]):
            t_batch = torch.full((x_0.shape[0],), t, device=device, dtype=x_0.dtype)
            score = score_fn(traj[i], t_batch)

            drift = sde.backward_drift(traj[i], t_batch, score_fn)
            diffusion = sde.diffusion(t_batch).view(-1, *([1] * (traj[i].ndim - 1)))
            drift_ode = drift - 0.5 * diffusion.pow(2) * score

            x = traj[i] + drift_ode * dt
            if controller is not None:
                x = controller.process_step(x_t=x, t=t_batch)
            traj[i + 1] = x

        return times, traj
