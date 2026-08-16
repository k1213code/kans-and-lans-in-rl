"""Track simple wall-clock timing during training."""

from __future__ import annotations

import time

from stable_baselines3.common.callbacks import BaseCallback


class TimeTrackingCallback(BaseCallback):
    """
    Log elapsed time, step rate, and rollout duration during training.
    """

    def __init__(self, log_every_n_calls: int = 100, verbose: int = 0):
        """
        Prepare the timers that will be initialized once training actually starts.
        """
        super().__init__(verbose)
        self.log_every_n_calls = log_every_n_calls

        # These only become meaningful after SB3 actually starts the run.
        self.training_start_time: float = 0.0
        self.last_log_time: float = 0.0
        self.last_log_timesteps: int = 0

        self.rollout_start_time: float = 0.0

    def _on_training_start(self) -> None:
        """
        Start the wall-clock timers right when SB3 enters the training loop.
        """
        # This is the cleanest place to start the wall-clock timers.
        now = time.perf_counter()
        self.training_start_time = now
        self.last_log_time = now
        self.last_log_timesteps = self.num_timesteps

    def _on_rollout_start(self) -> None:
        """
        Mark the beginning of a rollout so rollout duration can be measured later.
        """
        self.rollout_start_time = time.perf_counter()

    def _on_rollout_end(self) -> None:
        """
        Log how long the just-finished rollout took.
        """
        if self.rollout_start_time <= 0:
            return
        rollout_elapsed = time.perf_counter() - self.rollout_start_time
        self.logger.record("custom_time/rollout_elapsed_s", rollout_elapsed)

    def _on_step(self) -> bool:
        """
        Periodically log total elapsed time and the recent training rate.
        """
        # Timing every step would not add much value.
        if self.n_calls % self.log_every_n_calls != 0:
            return True

        now = time.perf_counter()

        # Compare the current wall-clock time and timestep count to the last log point.
        elapsed_total = now - self.training_start_time
        delta_time = now - self.last_log_time
        delta_steps = self.num_timesteps - self.last_log_timesteps

        self.logger.record("custom_time/elapsed_s", elapsed_total)
        self.logger.record("custom_time/elapsed_min", elapsed_total / 60.0)

        # Protect against divide-by-zero when the callback fires too quickly.
        if delta_time > 0:
            self.logger.record("custom_time/steps_per_second", delta_steps / delta_time)
            self.logger.record(
                "custom_time/seconds_per_1000_steps",
                (1000.0 * delta_time / delta_steps) if delta_steps > 0 else 0.0,
            )

        # Reset the reference point for the next timing sample.
        self.last_log_time = now
        self.last_log_timesteps = self.num_timesteps
        return True
