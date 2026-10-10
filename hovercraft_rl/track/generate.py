"""Gera as pistas padrão (curvas paramétricas) e grava em ``track/data/*.json``.

Uso: ``python -m hovercraft_rl.track.generate``
"""

from __future__ import annotations

import numpy as np

from .track import DATA_DIR, Track


def _resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Reamostra uma curva fechada densa com espaçamento ~uniforme (m)."""
    p = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 8)
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    closed = np.vstack([p, p[:1]])
    return np.stack([np.interp(targets, s, closed[:, 0]), np.interp(targets, s, closed[:, 1])], axis=1)


def _periodic_spline(ctrl: list[tuple[float, float]], samples: int = 4000) -> np.ndarray:
    """Spline Catmull-Rom periódica (centrípeta) passando pelos pontos de controle."""
    c = np.asarray(ctrl, dtype=np.float64)
    n = len(c)
    out = []
    per = samples // n
    for i in range(n):
        p0, p1, p2, p3 = c[(i - 1) % n], c[i], c[(i + 1) % n], c[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        t = np.linspace(t1, t2, per, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def _stadium(straight: float, radius: float) -> np.ndarray:
    """Oval: duas retas + duas semicircunferências; largada no meio da reta inferior."""
    pts = []
    xs = np.linspace(0, straight / 2, 200, endpoint=False)
    pts += [(x, -radius) for x in xs]
    th = np.linspace(-np.pi / 2, np.pi / 2, 400, endpoint=False)
    pts += [(straight / 2 + radius * np.cos(t), radius * np.sin(t)) for t in th]
    xs = np.linspace(straight / 2, -straight / 2, 400, endpoint=False)
    pts += [(x, radius) for x in xs]
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 400, endpoint=False)
    pts += [(-straight / 2 + radius * np.cos(t), radius * np.sin(t)) for t in th]
    xs = np.linspace(-straight / 2, 0, 200, endpoint=False)
    pts += [(x, -radius) for x in xs]
    return np.array(pts)


def _rounded_rect(w: float, h: float, rc: float) -> np.ndarray:
    pts = []
    corners = [(w / 2 - rc, -h / 2 + rc, -np.pi / 2), (w / 2 - rc, h / 2 - rc, 0.0),
               (-w / 2 + rc, h / 2 - rc, np.pi / 2), (-w / 2 + rc, -h / 2 + rc, np.pi)]
    pts.append((0.0, -h / 2))
    for cx, cy, a0 in corners:
        th = np.linspace(a0, a0 + np.pi / 2, 100)
        pts += [(cx + rc * np.cos(t), cy + rc * np.sin(t)) for t in th]
    return _resample_closed(np.array(pts), 0.05)


def build_tracks(spacing: float = 1.0) -> dict[str, Track]:
    th = np.linspace(0, 2 * np.pi, 4000, endpoint=False) - np.pi / 2
    circle = np.stack([30 * np.cos(th), 30 * np.sin(th) + 30], axis=1)

    r = 30 * (1 + 0.45 * np.cos(2 * (th + np.pi / 2)))  # "amendoim": duas retas côncavas
    kidney = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)

    gp_ctrl = [(0, 0), (40, 0), (70, 4), (82, 22), (72, 40), (52, 42), (44, 56),
               (54, 72), (48, 88), (28, 90), (18, 74), (4, 62), (-14, 70), (-30, 62),
               (-34, 40), (-22, 26), (-30, 10), (-20, 0)]
    gp = _periodic_spline(gp_ctrl)

    specs = {
        "circle": (circle, 3.0),
        "oval": (_stadium(60.0, 20.0), 3.0),
        "rounded_rect": (_rounded_rect(90.0, 50.0, 10.0), 3.0),
        "kidney": (kidney, 3.0),
        "gp": (gp, 3.0),
    }
    return {name: Track(_resample_closed(p, spacing), hw, name=name) for name, (p, hw) in specs.items()}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, tr in build_tracks().items():
        tr.save(DATA_DIR / f"{name}.json")
        rmin = tr.min_curvature_radius()
        flag = "" if rmin > tr.half_width else "  <-- ATENÇÃO: raio < meia-largura"
        print(f"{tr}  raio mín ≈ {rmin:.1f} m{flag}")


if __name__ == "__main__":
    main()
