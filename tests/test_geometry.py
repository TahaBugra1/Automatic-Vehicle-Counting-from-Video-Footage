import pytest

from visionflow.geometry import Line

LINE = Line((0, 100), (200, 100))


def test_flipped_swaps_start_and_end():
    assert LINE.flipped() == Line((200, 100), (0, 100))


def test_flipped_reverses_signed_distance_sign():
    p = (50, 130)
    assert LINE.flipped().signed_distance(p) == pytest.approx(-LINE.signed_distance(p))


def test_to_pixels_scales_normalized_coordinates():
    line = Line((0.25, 0.5), (0.75, 1.0)).to_pixels(800, 600)
    assert line == Line((200.0, 300.0), (600.0, 600.0))


def test_projection_is_zero_at_start_one_at_end_and_unclamped():
    assert LINE.projection((0, 100)) == pytest.approx(0.0)
    assert LINE.projection((200, 50)) == pytest.approx(1.0)
    assert LINE.projection((100, 999)) == pytest.approx(0.5)
    assert LINE.projection((-100, 100)) == pytest.approx(-0.5)
    assert LINE.projection((400, 100)) == pytest.approx(2.0)


def test_signed_distance_sign_and_magnitude():
    # Line points along +x; in image coordinates (y down) the positive side is y > 100.
    assert LINE.signed_distance((100, 130)) == pytest.approx(30.0)
    assert LINE.signed_distance((100, 70)) == pytest.approx(-30.0)
    assert LINE.signed_distance((100, 100)) == pytest.approx(0.0)
