import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RELIABILITY_PATH = Path(__file__).resolve().parents[2] / "config" / "reliability.json"


@dataclass(frozen=True, slots=True)
class ReliabilityThresholds:
    high_below: float
    medium_below: float


def load_reliability(path: Path | None = None) -> ReliabilityThresholds:
    data = json.loads((path or DEFAULT_RELIABILITY_PATH).read_text(encoding="utf-8"))
    return ReliabilityThresholds(float(data["high_below"]), float(data["medium_below"]))
