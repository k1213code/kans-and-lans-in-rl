"""Track how much RAM the training run uses."""

from __future__ import annotations

import os

import psutil
from stable_baselines3.common.callbacks import BaseCallback


class MemoryTrackingCallback(BaseCallback):
    """
    Log RAM usage of the training process and its child processes.
    """

    def __init__(self, log_every_n_calls: int = 100, verbose: int = 0):
        """
        Anchor memory tracking to the current training process tree.
        """
        super().__init__(verbose)
        self.log_every_n_calls = log_every_n_calls
        # Everything else is measured relative to this process tree.
        self.process = psutil.Process(os.getpid())
        self.peak_total_rss_mb = 0.0

    @staticmethod
    def _bytes_to_mb(num_bytes: int) -> float:
        """
        Convert raw byte counts into megabytes for easier logging.
        """
        return num_bytes / (1024 * 1024)

    def _collect_memory_stats(self) -> dict[str, float]:
        """
        Collect memory usage for the main process, children, and the running peak.
        """
        # Sample the main Python process first because that is always present.
        mem = self.process.memory_info()

        main_rss_mb = self._bytes_to_mb(mem.rss)

        children_rss_mb = 0.0
        children_vms_mb = 0.0

        # Add child processes so subprocess-backed env workers are not missed.
        for child in self.process.children(recursive=True):
            try:
                # In practice these are usually env workers or library helper processes.
                child_mem = child.memory_info()
                children_rss_mb += self._bytes_to_mb(child_mem.rss)
                children_vms_mb += self._bytes_to_mb(child_mem.vms)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Child processes can disappear between enumeration and sampling.
                continue

        total_rss_mb = main_rss_mb + children_rss_mb

        # Keep a simple running peak so the final logs show the worst observed memory usage.
        self.peak_total_rss_mb = max(self.peak_total_rss_mb, total_rss_mb)

        return {
            "memory/main_rss_mb": main_rss_mb,
            "memory/children_rss_mb": children_rss_mb,
            "memory/children_vms_mb": children_vms_mb,
            "memory/total_rss_mb": total_rss_mb,
            "memory/peak_total_rss_mb": self.peak_total_rss_mb,
        }

    def _on_step(self) -> bool:
        """
        Periodically log memory instead of sampling on every callback invocation.
        """
        # Sampling every callback call would be unnecessary noise.
        if self.n_calls % self.log_every_n_calls != 0:
            return True

        stats = self._collect_memory_stats()
        for key, value in stats.items():
            self.logger.record(key, value)

        return True
