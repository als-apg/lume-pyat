# lume-pyat

pyAT-specific implementation of [LUME](https://github.com/lume-science) model
classes for virtual accelerators.

`lume-pyat` sits between [lume-base](https://github.com/lume-science/lume-base)
and [pyAT](https://github.com/atcollab/at). You build the lattice; this package
wraps it in a `PyATSimulator`, binds free-form variable names to element
attributes in native pyAT units, and exposes the result as a lume-base
`ActionModel`. It is a sibling of
[lume-cheetah](https://github.com/lume-science/lume-cheetah) and follows the
same conventions.

The package is facility-agnostic. It ships no lattice data and knows nothing
about any particular machine's naming, units, or calibration. Mapping
control-system addresses and engineering units onto simulator parameters is the
job of the layer above — and `PyATWritableScalarVariable` is the seam where
that layer plugs in.

## Installation

```bash
pip install lume-pyat
```

Requires Python 3.10 or newer.

## What it gives you

- **Atomic multi-set.** A batch of writes either lands completely or not at
  all. Every element the batch touches is snapshotted first, the whole batch is
  applied, the orbit is solved **once**, and only then is anything committed. A
  failure restores every snapshot, so a rejected write is a complete no-op.
- **Instability is an exception, not a NaN.** pyAT reports an unstable ring by
  returning non-finite values rather than raising. `solve_orbit` checks the
  one-turn matrix and closed orbit by value and raises `OrbitSolveError`, so a
  caller cannot serve garbage as a monitor reading.
- **One persistent lattice.** It is mutated in place, never copied or rebuilt.
  Writes compose the way their physical counterparts do, and seeded
  misalignments survive every later write.
- **Cheap imports.** `import lume_pyat` pulls in neither pyAT nor lume-base nor
  HDF5 — resolution is lazy, so error paths and light entry points stay light.

## Example

```python
import at
import numpy as np

from lume_pyat import (
    LUMEPyATModel,
    PyATReadOnlyScalarVariable,
    PyATSimulator,
    PyATWritableScalarVariable,
)

# 1. Build a lattice. lume-pyat never builds one for you -- this is a small
#    FODO ring, but any 4D-canonical at.Lattice works.
N_CELLS = 8
elements = []
for cell in range(1, N_CELLS + 1):
    tag = f"{cell:02d}"
    elements += [
        at.Monitor(f"BPM_{tag}"),
        at.Drift("DRIFT", 0.4),
        at.Quadrupole(f"QUAD_F_{tag}", 0.3, 1.0),
        at.Drift("DRIFT", 0.4),
        at.Corrector(f"COR_H_{tag}", 0.0, [0.0, 0.0]),
        at.Dipole(f"BEND_{tag}", 1.0, 2 * np.pi / N_CELLS),
        at.Drift("DRIFT", 0.4),
        at.Quadrupole(f"QUAD_D_{tag}", 0.3, -1.0),
        at.Drift("DRIFT", 0.4),
    ]
ring = at.Lattice(elements, name="EXAMPLE", energy=1.0e9, periodicity=1)
ring.disable_6d()

# 2. Hand the lattice to a simulator. It is mutated in place, never copied.
simulator = PyATSimulator(ring)

# 3. Bind names to element attributes. Names are free-form -- use whatever your
#    own control system calls these. Values are in native pyAT units.
variables = [
    PyATWritableScalarVariable(
        name="corrector_kick",
        element_name="COR_H_01",
        attribute="KickAngle",
        index=0,
        default_value=0.0,
        unit="rad",
    ),
    *(
        PyATReadOnlyScalarVariable(
            name=f"bpm_{cell:02d}_x", element_name=f"BPM_{cell:02d}", axis="x", unit="m"
        )
        for cell in range(1, N_CELLS + 1)
    ),
]

# 4. The model solves the boot orbit once and caches it.
model = LUMEPyATModel(simulator=simulator, action_variables=variables)

bpm_names = [f"bpm_{cell:02d}_x" for cell in range(1, N_CELLS + 1)]
print(f"{len(model.supported_variables)} variables: 1 input, {len(bpm_names)} outputs")

# 5. Read: the orbit is flat and the corrector sits at its default.
print(f"before:  corrector = {model.get('corrector_kick')} rad")
print(f"         max |x|   = {max(abs(v) for v in model.get(bpm_names).values()):.3e} m")

# 6. Write: one batch, one solve, all or nothing.
model.set({"corrector_kick": 1.0e-4})
print(f"after:   corrector = {model.get('corrector_kick')} rad   <- retained exactly")
print(f"         max |x|   = {max(abs(v) for v in model.get(bpm_names).values()):.3e} m")

# 7. A write that leaves the ring without a stable orbit is a complete no-op.
try:
    model.set({"corrector_kick": 1.0})
except Exception as exc:
    print(f"rejected: {type(exc).__name__} -- corrector still {model.get('corrector_kick')}")

# 8. reset() puts every input back to its default, in one atomic batch.
model.reset()
print(f"reset:   corrector = {model.get('corrector_kick')} rad")
print(f"         max |x|   = {max(abs(v) for v in model.get(bpm_names).values()):.3e} m")

assert model.get("corrector_kick") == 0.0
assert max(abs(v) for v in model.get(bpm_names).values()) < 1e-12
```

Which prints:

```
9 variables: 1 input, 8 outputs
before:  corrector = 0.0 rad
         max |x|   = 0.000e+00 m
after:   corrector = 0.0001 rad   <- retained exactly
         max |x|   = 8.155e-04 m
rejected: OrbitSolveError -- corrector still 0.0001
reset:   corrector = 0.0 rad
         max |x|   = 0.000e+00 m
```

The example code above is extracted and executed on every CI run, so it cannot
drift from the package. (The sample output is illustrative — the last digits
depend on the platform's linear algebra.)

## Adding your own units

`PyATWritableScalarVariable` writes native pyAT quantities. To control an
element in your own units, subclass it and put the conversion in `_set` and
`_get`:

```python
class CurrentVariable(PyATWritableScalarVariable):
    amps_per_unit: float

    def _set(self, simulator, value):
        super()._set(simulator, value / self.amps_per_unit)

    def _get(self, simulator):
        return super()._get(simulator) * self.amps_per_unit
```

The conversion belongs to your layer, not to this package. Everything else —
the atomicity, the solve, the caching — keeps working unchanged.

## API

| Name | What it is |
|------|-----------|
| `PyATSimulator` | Owns one `at.Lattice`; `solve()`, `element()`, `element_index()`, `lattice`, `last_solution` |
| `LUMEPyATModel` | `ActionModel` over a simulator; atomic `set()`, cached `get()`, batched `reset()` |
| `PyATWritableScalarVariable` | Binds a name to one element attribute; the extension point |
| `PyATReadOnlyScalarVariable` | One transverse coordinate of a monitor's reading |
| `solve_orbit`, `monitor_xy` | Guarded 4D closed-orbit solve and its monitor readout |
| `apply_misalignment` | dx/dy/roll element misalignment, ported from pySC |
| `OrbitSolveError`, `UnknownElementError` | The exception contract |

## Development

```bash
uv sync --extra dev
uv run pytest lume_pyat/tests
uv run pre-commit run --all-files
```

## License

Apache-2.0. See [LICENSE](LICENSE).
