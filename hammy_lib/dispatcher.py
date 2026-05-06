"""Queue-based worker dispatch for Task execution.

CPU workers take the smallest tasks; GPU workers take the largest.
"""
from __future__ import annotations
from collections import deque

from .task import Task


class Dispatcher:
    """Dispatch tasks to CPU/GPU workers based on graph size.

    Smaller graphs → CPU (lower overhead).
    Larger graphs → GPU (throughput advantage).
    """

    def __init__(self, tasks: list[Task]):
        self.queue: deque[Task] = deque(sorted(tasks, key=lambda t: t.size))

    def __len__(self) -> int:
        return len(self.queue)

    def next_for_cpu(self) -> Task:
        """Pop the smallest available task."""
        if not self.queue:
            raise IndexError("No tasks remaining")
        return self.queue.popleft()

    def next_for_gpu(self) -> Task:
        """Pop the largest available task."""
        if not self.queue:
            raise IndexError("No tasks remaining")
        return self.queue.pop()

    def has_tasks(self) -> bool:
        return bool(self.queue)
