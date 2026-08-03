"""Import weight: what pulling in this package does, and does not, cost.

Every check runs in a subprocess. Measuring ``sys.modules`` in-process would
be meaningless — the test session has already imported everything.
"""

import subprocess
import sys

import pytest

# Importing the package must not drag in the simulator, pyAT, the lume-base
# model machinery, HDF5, or numpy. Consumers re-export these names on error
# paths and from lightweight entry points, and numpy belongs on the list for
# the same reason as the rest: the top-level namespace resolves lazily, so
# nothing heavier than the standard library has any business being imported.
HEAVY_ROOTS = ("at", "h5py", "lume", "numpy")

# The exceptions module carries the same bar, plus scipy for good measure.
STDLIB_ONLY_ROOTS = (*HEAVY_ROOTS, "scipy")


def imported_roots(statement: str, roots: tuple[str, ...]) -> list[str]:
    """Roots from ``roots`` present in sys.modules after running ``statement``."""
    probe = (
        f"{statement}; "
        "import sys; "
        f"roots = {roots!r}; "
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules} & set(roots))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return [root for root in result.stdout.strip().split(",") if root]


def test_importing_the_package_is_cheap():
    assert imported_roots("import lume_pyat", HEAVY_ROOTS) == []


def test_importing_the_exceptions_module_is_standard_library_only():
    assert imported_roots("import lume_pyat.exceptions", STDLIB_ONLY_ROOTS) == []


@pytest.mark.parametrize(
    "name",
    ["OrbitSolveError", "UnknownElementError"],
)
def test_reaching_an_exception_through_the_package_stays_cheap(name):
    # The lazy top-level namespace must resolve these without waking anything
    # else: this is the path a consumer's own re-export shim takes.
    statement = f"import lume_pyat; lume_pyat.{name}"
    assert imported_roots(statement, STDLIB_ONLY_ROOTS) == []


def test_the_probe_detects_a_heavy_import():
    # Positive control: without this, a probe that silently measured nothing
    # would make every check above pass for the wrong reason.
    assert imported_roots("import lume_pyat.model", HEAVY_ROOTS) == [
        "at",
        "h5py",
        "lume",
        "numpy",
    ]
    # The simulator reaches pyAT but not lume-base -- only the model and the
    # action variables depend on the contract layer.
    assert imported_roots("import lume_pyat.simulator", HEAVY_ROOTS) == [
        "at",
        "h5py",
        "numpy",
    ]
