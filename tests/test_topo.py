import numpy as np
import pytest

from msvae.topo import TopoProjector, _calc_g, _legendre_poly_values


def _positions(n=48, seed=0):
    """Positions bien reparties sur une calotte spherique (spirale de Fibonacci).

    Une repartition reguliere est indispensable : deux electrodes tres proches
    rendent le systeme spline mal conditionne, ce qui n'a rien a voir avec la
    qualite de l'interpolation qu'on veut tester.
    """
    i = np.arange(n) + 0.5
    theta = np.arccos(1 - 0.75 * i / n)          # calotte : theta < ~1.3 rad
    phi = np.pi * (1 + 5 ** 0.5) * i
    return 0.09 * np.stack([np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)], axis=1)


def test_legendre_recurrence():
    x = np.linspace(-1, 1, 11)
    p = _legendre_poly_values(x, 4)
    np.testing.assert_allclose(p[0], x, atol=1e-12)
    np.testing.assert_allclose(p[1], 0.5 * (3 * x ** 2 - 1), atol=1e-12)
    np.testing.assert_allclose(p[2], 0.5 * (5 * x ** 3 - 3 * x), atol=1e-12)


def test_green_function_decreases_with_angle():
    g = _calc_g(np.array([1.0, 0.5, 0.0, -1.0]))
    assert g[0] > g[1] > g[2] > g[3]


def test_projection_inside_unit_disk():
    pos = _positions()
    tp = TopoProjector.from_positions(pos, [f"E{i}" for i in range(len(pos))], size=32)
    assert np.linalg.norm(tp.xy, axis=1).max() <= 1.0
    assert tp.mask.sum() > 0.7 * 32 ** 2 * np.pi / 4


def test_image_is_linear_in_data():
    pos = _positions()
    tp = TopoProjector.from_positions(pos, [f"E{i}" for i in range(len(pos))], size=32)
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, len(pos))).astype(np.float32)
    np.testing.assert_allclose(tp.to_image(a + 2 * b),
                               tp.to_image(a) + 2 * tp.to_image(b),
                               atol=1e-4, rtol=1e-3)


def test_exact_roundtrip_on_generated_images():
    """La representation en image doit etre (quasi) sans perte."""
    pos = _positions()
    tp = TopoProjector.from_positions(pos, [f"E{i}" for i in range(len(pos))], size=32)
    rng = np.random.default_rng(2)
    v = rng.standard_normal((50, len(pos))).astype(np.float32)
    back = tp.to_topo(tp.to_image(v), method="pinv")
    r = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(v, back)])
    assert r > 0.99


def test_smooth_field_interpolation_is_faithful():
    pos = _positions()
    tp = TopoProjector.from_positions(pos, [f"E{i}" for i in range(len(pos))], size=32)
    v = pos[:, 1] / pos[:, 1].std()  # champ lisse (gradient antero-posterieur)
    back = tp.to_topo(tp.to_image(v.astype(np.float32)), method="ridge")
    assert np.corrcoef(v, back)[0, 1] > 0.98


@pytest.mark.parametrize("method", ["ridge", "pinv", "bilinear"])
def test_to_topo_methods_shapes(method):
    pos = _positions()
    tp = TopoProjector.from_positions(pos, [f"E{i}" for i in range(len(pos))], size=32)
    img = tp.to_image(np.ones((3, len(pos)), dtype=np.float32))
    assert tp.to_topo(img, method=method).shape == (3, len(pos))
