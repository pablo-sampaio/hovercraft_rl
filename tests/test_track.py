import numpy as np
import pytest

from hovercraft_rl.track import Track, available_tracks


@pytest.fixture
def square():
    # quadrado 10x10 anti-horário, largada em (0, 0)
    return Track([[0, 0], [10, 0], [10, 10], [0, 10]], half_width=1.0, name="square")


def test_length_and_point_at(square):
    assert square.length == pytest.approx(40.0)
    np.testing.assert_allclose(square.point_at(15.0), [10.0, 5.0])
    np.testing.assert_allclose(square.point_at(41.0), [1.0, 0.0])
    assert square.heading_at(15.0) == pytest.approx(np.pi / 2)


def test_projection_on_segment_not_vertex(square):
    p = square.project(3.3, 0.4)
    assert (p.x_ref, p.y_ref) == pytest.approx((3.3, 0.0))
    assert p.s == pytest.approx(3.3)
    assert p.lateral_error == pytest.approx(0.4)    # à esquerda (dentro do quadrado)
    p = square.project(3.3, -0.7)
    assert p.lateral_error == pytest.approx(-0.7)


def test_progress_wraps(square):
    assert square.progress_delta(39.5, 0.5) == pytest.approx(1.0)
    assert square.progress_delta(0.5, 39.5) == pytest.approx(-1.0)


def test_windowed_search_avoids_jumps():
    # pista estreita em "U": dois trechos paralelos a 2 m um do outro
    pts = [[0, 0], [50, 0], [50, 2], [0, 2], [-5, 1]]
    tr = Track(pts, half_width=0.9, search_window=5.0)
    # ponto mais próximo globalmente do trecho de cima, mas com hint no trecho de baixo
    p = tr.project(25.0, 1.2, hint_s=25.0)
    assert p.y_ref == pytest.approx(0.0)
    assert abs(p.lateral_error) > tr.half_width   # => fora da pista, sem "atalho"
    assert tr.project(25.0, 1.2).y_ref == pytest.approx(2.0)


@pytest.mark.parametrize("name", available_tracks())
def test_saved_tracks_are_valid(name):
    tr = Track.load(name)
    assert tr.length > 50
    assert tr.min_curvature_radius() > tr.half_width
