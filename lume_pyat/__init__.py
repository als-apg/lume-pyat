"""LUME model and simulator classes backed by pyAT lattices.

Attribute access is lazy (PEP 562): importing :mod:`lume_pyat` pulls in no
simulator, model, or file-format machinery. Nothing heavier than the standard
library is imported until a public name is actually used.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import shapes for type checkers only
    from lume_pyat.actions import (
        PyATReadOnlyScalarVariable,
        PyATWritableScalarVariable,
    )
    from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
    from lume_pyat.model import LUMEPyATModel
    from lume_pyat.simulator import (
        ElementState,
        PyATSimulator,
        restore_element,
        snapshot_element,
    )
    from lume_pyat.solve import monitor_xy, solve_orbit
    from lume_pyat.utils import apply_misalignment

_LAZY_NAMES = {
    "ElementState": "lume_pyat.simulator",
    "LUMEPyATModel": "lume_pyat.model",
    "OrbitSolveError": "lume_pyat.exceptions",
    "PyATReadOnlyScalarVariable": "lume_pyat.actions",
    "PyATSimulator": "lume_pyat.simulator",
    "PyATWritableScalarVariable": "lume_pyat.actions",
    "UnknownElementError": "lume_pyat.exceptions",
    "apply_misalignment": "lume_pyat.utils",
    "monitor_xy": "lume_pyat.solve",
    "restore_element": "lume_pyat.simulator",
    "snapshot_element": "lume_pyat.simulator",
    "solve_orbit": "lume_pyat.solve",
}

# Kept as a literal so linters and type checkers can see the public surface
# even though every name behind it is resolved lazily.
__all__ = [
    "ElementState",
    "LUMEPyATModel",
    "OrbitSolveError",
    "PyATReadOnlyScalarVariable",
    "PyATSimulator",
    "PyATWritableScalarVariable",
    "UnknownElementError",
    "__version__",
    "apply_misalignment",
    "monitor_xy",
    "restore_element",
    "snapshot_element",
    "solve_orbit",
]


def _read_version() -> str:
    try:
        from lume_pyat._version import version
    except ImportError:  # not built — e.g. a plain source checkout
        return "0.0.0+unknown"
    return version


def __getattr__(name: str):
    if name == "__version__":
        return _read_version()
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
