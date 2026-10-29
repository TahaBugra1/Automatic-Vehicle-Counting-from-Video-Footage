"""Ground-truth line crossings from UA-DETRAC annotations, and frame folders to video.

Assumed annotation layout (UA-DETRAC "DETRAC-Train/Test-Annotations-XML"):

    <sequence name="MVI_20011">
      <sequence_attribute .../>                                  (optional, unused)
      <ignored_region><box left top width height/>...</ignored_region>  (optional)
      <frame density="7" num="1">                                (num is 1-based)
        <target_list>
          <target id="1">
            <box left="592.75" top="378.8" width="160.05" height="162.2"/>
            <attribute vehicle_type="car" .../>                  (optional, unused)
          </target>
        </target_list>
      </frame>
    </sequence>
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visionflow.counter import LineCounter
from visionflow.geometry import Line
from visionflow.models import Detection, Direction, VehicleClass

UA_DETRAC_FPS = 25.0

Box = tuple[float, float, float, float]  # left, top, width, height


@dataclass(frozen=True, slots=True)
class GtCrossing:
    frame: int  # 0-based, like the pipeline
    track_id: int
    direction: Direction


def _number(elem: ET.Element, attr: str, path: Path) -> float:
    raw = elem.get(attr)
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"malformed UA-DETRAC annotation {path.name}: <{elem.tag}> has invalid {attr}={raw!r}"
        ) from None


def _box(elem: ET.Element, path: Path) -> Box:
    return tuple(_number(elem, a, path) for a in ("left", "top", "width", "height"))


def _inside(point: tuple[float, float], box: Box) -> bool:
    left, top, width, height = box
    return left <= point[0] <= left + width and top <= point[1] <= top + height


def _parse(xml_path: Path) -> ET.Element:
    try:
        return ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"malformed UA-DETRAC annotation {xml_path.name}: {exc}") from None


def _ignored(root: ET.Element, xml_path: Path) -> list[Box]:
    return [_box(b, xml_path) for r in root.iter("ignored_region") for b in r.iter("box")]


def load_ignored_regions(xml_path: Path) -> list[Box]:
    """The sequence's ignored-region boxes (left, top, width, height) in pixels."""
    return _ignored(_parse(xml_path), xml_path)


def segment_hits_box(line: Line, box: Box) -> bool:
    """True when the pixel segment `line` touches the box (Liang-Barsky clipping)."""
    left, top, width, height = box
    (x0, y0), (x1, y1) = line.start, line.end
    dx, dy = x1 - x0, y1 - y0
    t_in, t_out = 0.0, 1.0
    edges = (
        (-dx, x0 - left),
        (dx, left + width - x0),
        (-dy, y0 - top),
        (dy, top + height - y0),
    )
    for p, q in edges:
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t_in = max(t_in, t)
        else:
            t_out = min(t_out, t)
        if t_in > t_out:
            return False
    return True


def load_annotations(xml_path: Path) -> list[Detection]:
    """One Detection per (frame, target), frame indices 0-based, ignored regions removed."""
    root = _parse(xml_path)
    ignored = _ignored(root, xml_path)
    detections: list[Detection] = []
    for frame_elem in root.iter("frame"):
        frame = int(_number(frame_elem, "num", xml_path)) - 1
        for target in frame_elem.iter("target"):
            box_elem = target.find("box")
            if box_elem is None:
                continue
            track_id = int(_number(target, "id", xml_path))
            left, top, width, height = _box(box_elem, xml_path)
            bbox = (left, top, left + width, top + height)
            det = Detection(
                frame, frame / UA_DETRAC_FPS, track_id, VehicleClass.car, 1.0, bbox
            )
            if not any(_inside(det.bottom_center, region) for region in ignored):
                detections.append(det)
    detections.sort(key=lambda d: (d.frame, d.track_id))
    return detections


def derive_ground_truth_crossings(
    xml_path: Path, line: Line, width: int, height: int
) -> list[GtCrossing]:
    """Crossings of the annotated tracks, using exactly the system's LineCounter rule."""
    counter = LineCounter(line.to_pixels(width, height))
    crossings: list[GtCrossing] = []
    for det in load_annotations(xml_path):
        event = counter.update(det)
        if event is not None:
            crossings.append(GtCrossing(event.frame, event.track_id, event.direction))
    return crossings


def frames_to_video(frames_dir: Path, out_path: Path, fps: float = UA_DETRAC_FPS) -> Path:
    """Write the sequence's img*.jpg frames (sorted by name) to an mp4v video."""
    images = sorted(frames_dir.glob("img*.jpg"))
    if not images:
        raise ValueError(f"no img*.jpg frames in {frames_dir} (missing or empty folder)")
    writer = None
    try:
        for image_path in images:
            # imdecode + fromfile: cv2.imread cannot open non-ASCII paths on Windows.
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read frame {image_path}")
            if writer is None:
                size = (image.shape[1], image.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
                if not writer.isOpened():
                    raise ValueError(f"cannot write video {out_path}")
            if (image.shape[1], image.shape[0]) != size:
                raise ValueError(f"frame {image_path.name} has a different size")
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
    return out_path
