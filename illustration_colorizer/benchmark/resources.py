from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class ResourceUsage:
    peak_cpu_rss_bytes: int | None
    peak_gpu_memory_bytes: int | None


class ResourceMonitor:
    def __init__(self, poll_interval_seconds: float = 0.05) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._peak_cpu_rss_bytes: int | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceMonitor":
        self._running = True
        self._peak_cpu_rss_bytes = None

        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if psutil is not None:
            self._thread = threading.Thread(target=self._poll_cpu_memory, daemon=True)
            self._thread.start()

        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_seconds * 4)

    def _poll_cpu_memory(self) -> None:
        if psutil is None:
            return

        process = psutil.Process()
        while self._running:
            rss = int(process.memory_info().rss)
            if self._peak_cpu_rss_bytes is None or rss > self._peak_cpu_rss_bytes:
                self._peak_cpu_rss_bytes = rss
            time.sleep(self._poll_interval_seconds)

    def snapshot(self) -> ResourceUsage:
        peak_gpu_memory_bytes: int | None = None
        if torch is not None and torch.cuda.is_available():
            peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())

        return ResourceUsage(
            peak_cpu_rss_bytes=self._peak_cpu_rss_bytes,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        )
