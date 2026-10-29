from dataclasses import dataclass
from enum import StrEnum


class VehicleClass(StrEnum):
    car = "car"
    bus = "bus"
    truck = "truck"
    motorcycle = "motorcycle"


class Direction(StrEnum):
    ONE = "Yön 1"
    TWO = "Yön 2"


@dataclass(frozen=True, slots=True)
class Detection:
    frame: int
    time_s: float
    track_id: int
    cls: VehicleClass
    confidence: float
    bbox: tuple[float, float, float, float]  # pixels: x1, y1, x2, y2

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    frame: int
    time_s: float
    track_id: int
    cls: VehicleClass
    direction: Direction
    confidence: float
