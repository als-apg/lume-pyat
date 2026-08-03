"""Package-level guarantees: the lazy surface and the shared test ring.

Import weight is covered separately, in test_imports.py.
"""

from pathlib import Path

import at
import numpy as np
import pytest

import lume_pyat
from lume_pyat.tests.conftest import QUAD_K, build_test_ring


def test_lazy_map_and_public_surface_agree():
    for module in lume_pyat._LAZY_NAMES.values():
        assert module.startswith("lume_pyat.")
    assert set(lume_pyat._LAZY_NAMES) | {"__version__"} == set(lume_pyat.__all__)
    assert lume_pyat.__all__ == sorted(lume_pyat.__all__)


def test_every_public_name_resolves():
    for name in lume_pyat.__all__:
        assert getattr(lume_pyat, name) is not None


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = lume_pyat.nope


def test_version_is_readable():
    assert isinstance(lume_pyat.__version__, str)


def test_py_typed_is_present():
    assert (Path(lume_pyat.__file__).parent / "py.typed").is_file()


def test_ring_is_stable_in_both_planes(test_ring):
    m44, _ = test_ring.find_m44()
    assert abs(np.trace(m44[:2, :2])) < 2.0
    assert abs(np.trace(m44[2:, 2:])) < 2.0


def test_ring_exposes_the_generic_element_families(test_ring):
    for cls, count in (
        (at.Monitor, 8),
        (at.Quadrupole, 16),
        (at.Sextupole, 16),
        (at.Corrector, 16),
        (at.Dipole, 8),
    ):
        assert len(test_ring.get_uint32_index(cls)) == count


def test_each_built_ring_is_independent():
    # The simulator mutates its lattice in place, so every caller needs its own.
    first, second = build_test_ring(), build_test_ring()
    index = first.get_uint32_index("QUAD_F_01")[0]
    first[index].K = 2.5
    assert second[index].K == pytest.approx(QUAD_K)
