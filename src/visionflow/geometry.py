import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Line:
    """Directed line from start to end.

    The positive side is where signed_distance > 0 (the side the arrow points to).
    """

    start: tuple[float, float]
    end: tuple[float, float]

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("Line start and end must differ")

    def to_pixels(self, width: int, height: int) -> "Line":
        return Line(
            (self.start[0] * width, self.start[1] * height),
            (self.end[0] * width, self.end[1] * height),
        )

    def flipped(self) -> "Line":
        return Line(self.end, self.start)

    def _vector(self) -> tuple[float, float]:
        return (self.end[0] - self.start[0], self.end[1] - self.start[1])

    def signed_distance(self, p: tuple[float, float]) -> float:
        dx, dy = self._vector()
        cross = dx * (p[1] - self.start[1]) - dy * (p[0] - self.start[0])
        return cross / math.hypot(dx, dy)

    def projection(self, p: tuple[float, float]) -> float:
        """Position of p along the segment: 0 at start, 1 at end (unclamped)."""
        dx, dy = self._vector()
        dot = dx * (p[0] - self.start[0]) + dy * (p[1] - self.start[1])
        return dot / (dx * dx + dy * dy)
