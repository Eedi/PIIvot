from typing import Mapping, Optional
import wandb
from abc import abstractmethod

class Tracker:
    def __init__(self):
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Log metrics to result store.

        :param metrics: The metrics to log.
        :param step: An optional step to attach the metrics to (e.g. the epoch).
        :param prefix: An optional prefix to prepend to every key in metrics.
        """

    @abstractmethod
    def end_run(self) -> None:
        """
        End a run, has to be run after the c
        Returns:

        """

    # @abstractmethod
    # def add_scaler(self, tag: str, value: float, global_step: int):
    #     raise NotImplementedError

class WandbTracker(Tracker):
    def __init__(self, experiment_name: str, args: dict, resume: bool = False):
        super().__init__()
        wandb.init(
            # track hyperparameters and run metadata
            name=str(experiment_name),
            config=args,
            reinit=resume,
        )

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """
        logs metrics to wandb
        Args:
            metrics: a mapping of metric tags to values
            step: current step of epoch
            prefix:
        Returns:
            None
        """
        wandb.log(metrics, step=step)

    def end_run(self):
        wandb.finish()