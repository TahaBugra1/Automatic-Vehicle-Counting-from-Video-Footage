from .geometry import Line
from .models import CrossingEvent, Detection, Direction

DEADBAND_PX = 2.0
PROJECTION_MARGIN = 0.05


class LineCounter:
    """Counts each track at most once when it crosses a directed line.

    Direction.ONE is a move from the negative to the positive side
    (signed_distance > 0). Points within `deadband` pixels of the line are
    ignored, and crossings whose projection lies outside the segment
    (extended by `projection_margin`) are not counted.
    """

    def __init__(
        self,
        line: Line,
        deadband: float = DEADBAND_PX,
        projection_margin: float = PROJECTION_MARGIN,
    ):
        self.line = line
        self.deadband = deadband
        self.projection_margin = projection_margin
        self._side: dict[int, int] = {}
        self._counted: set[int] = set()
        self._counts: dict[Direction, int] = {d: 0 for d in Direction}

    @property
    def counts(self) -> dict[Direction, int]:
        return dict(self._counts)

    def update(self, det: Detection) -> CrossingEvent | None:
        point = det.bottom_center
        dist = self.line.signed_distance(point)
        if abs(dist) <= self.deadband:
            return None
        side = 1 if dist > 0 else -1
        previous = self._side.get(det.track_id)
        self._side[det.track_id] = side
        if previous is None or previous == side or det.track_id in self._counted:
            return None
        projection = self.line.projection(point)
        if not (-self.projection_margin <= projection <= 1 + self.projection_margin):
            return None
        self._counted.add(det.track_id)
        direction = Direction.ONE if side > 0 else Direction.TWO
        self._counts[direction] += 1
        return CrossingEvent(
            frame=det.frame,
            time_s=det.time_s,
            track_id=det.track_id,
            cls=det.cls,
            direction=direction,
            confidence=det.confidence,
        )
