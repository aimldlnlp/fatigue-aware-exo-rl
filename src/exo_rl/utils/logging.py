from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricTracker:
    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.values.setdefault(key, []).append(float(value))

    def mean(self, key: str, window: int | None = None) -> float:
        entries = self.values.get(key, [])
        if not entries:
            return 0.0
        if window is not None:
            entries = entries[-window:]
        return sum(entries) / len(entries)

