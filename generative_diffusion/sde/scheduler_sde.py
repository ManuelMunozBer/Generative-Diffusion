# sde/scheduler_sde.py
from __future__ import annotations

from abc import abstractmethod

from torch import Tensor

from .base_sde import BaseSDE
from generative_diffusion.schedulers import BaseScheduler


class MissingSchedulerError(ValueError):
    """Se lanza si la SDE requiere un scheduler y no se le pasa uno."""

    def __init__(self, cls):
        super().__init__(f"La SDE {cls.__name__} requiere un scheduler.")


class SchedulerBasedSDE(BaseSDE):
    """
    SDE cuya β(t) depende de un objeto *scheduler*.
    """

    def __init__(self, scheduler: BaseScheduler) -> None:
        if scheduler is None:
            raise MissingSchedulerError(self.__class__)
        super().__init__()
        self.scheduler = scheduler

    # ------------------------------------------------------------------ #
    @abstractmethod
    def beta_t(self, t: Tensor) -> Tensor:
        """β(t) vía scheduler."""
        raise NotImplementedError
