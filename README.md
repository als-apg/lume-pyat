# lume-pyat

pyAT-specific implementation of [LUME](https://github.com/lume-science) model
classes for virtual accelerators.

`lume-pyat` sits between [lume-base](https://github.com/lume-science/lume-base)
and [pyAT](https://github.com/atcollab/at). You build the lattice, 4D or 6D;
this package wraps it in a `PyATSimulator`, binds free-form variable names to
element attributes — or to the ring energy — in native pyAT units, and exposes
the result as a lume-base `ActionModel`. It is a sibling of
[lume-cheetah](https://github.com/lume-science/lume-cheetah) and follows the
same conventions.

The package is facility-agnostic. It ships no lattice data and knows nothing
about any particular machine's naming, units, or calibration. Mapping
control-system addresses and engineering units onto simulator parameters is the
job of the layer above — and `PyATWritableScalarVariable` and
`PyATLatticeScalarVariable` are the seams where that layer plugs in.

## Installation

```bash
pip install lume-pyat
```

Requires Python 3.10 or newer.

## What it gives you

- **Atomic multi-set.** A batch of writes either lands completely or not at
  all. Everything the batch touches — every bound element, and for a
  lattice-level variable the ring energy and every cavity — is snapshotted
  first, the whole batch is applied, the orbit is solved **once**, and only
  then is anything committed. A failure restores every snapshot, so a rejected
  write is a complete no-op.
- **One setpoint, several elements.** A writable variable binds a list of
  elements, each with a weight: a write puts `value × weight` on every one of
  them, and a read comes from the first. A magnet sliced into several lattice
  elements, or a kick shared across them, is one variable.
- **The ring energy is a variable too.** `PyATLatticeScalarVariable` sets
  `ring.energy` and every cavity's `Energy` as one atomic write, with a hook
  for the layer above to rescale strengths in the same batch.
- **Bindings are checked where you declare them.** A variable naming an element
  the lattice does not have, a name two elements share, an attribute that does
  not exist, or an index past the end of one is rejected when the model is
  built — not at whichever later write happens to touch it.
- **Instability is an exception, not a NaN.** pyAT reports an unstable ring by
  returning non-finite values rather than raising — or, with a cavity in the
  ring, by returning a finite one-turn matrix whose eigenvalues have left the
  unit circle. `solve_orbit` checks the one-turn matrix's eigenvalues in all
  three planes and the closed orbit by value, and raises `OrbitSolveError`
  naming the unstable plane, so a caller cannot serve garbage as a monitor
  reading.
- **4D or 6D.** A ring without a cavity is solved in 4D, a ring with one
  enabled in 6D, so RF-driven dispersive orbits solve as they should.
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

## Several elements from one setpoint

The `element_name=` / `attribute=` / `index=` form above is shorthand for a
single `ElementBinding` with unit weight. Give `bindings=` instead to drive
several elements from one variable. A write of `v` puts `v × weight` on every
bound element; a read returns the first element's value divided by its weight.

```python
from lume_pyat import ElementBinding, PyATWritableScalarVariable

# A corrector split over three lattice slices: each slice carries a third of
# the kick, and the variable reads back the whole kick from the first slice.
split_kick = PyATWritableScalarVariable(
    name="corrector_kick",
    bindings=[
        ElementBinding(element_name=name, attribute="KickAngle", index=0, weight=1 / 3)
        for name in ("COR_H_01A", "COR_H_01B", "COR_H_01C")
    ],
    default_value=0.0,
    unit="rad",
)

# A quadrupole sliced into two elements that share one strength.
sliced_quad = PyATWritableScalarVariable(
    name="quad_strength",
    bindings=[
        ElementBinding(element_name="QUAD_01A", attribute="K"),
        ElementBinding(element_name="QUAD_01B", attribute="K"),
    ],
    default_value=1.0,
)
```

Every binding is checked when the model is built, and a failed batch restores
every bound element.

## The ring energy

`PyATLatticeScalarVariable` binds the lattice as a whole rather than an
element. Its value is `ring.energy` in eV; a write sets that and the `Energy`
of every `at.RFCavity`, and a failed batch restores all of them.

Changing the energy does not change the normalised strengths pyAT stores. A
layer that holds magnet *currents* fixed across an energy change — so that
`K` falls as `1/E` — does the rescaling itself in the `_after_write` hook,
which runs inside the same batch as the energy write, and declares what it
touches in `snapshot_targets` so the rescale rolls back with the energy:

```python
class EnergyFromDipoleCurrent(PyATLatticeScalarVariable):
    ev_per_amp: float
    reference_energy: float  # the energy the reference strengths belong to, in eV
    reference_strengths: dict[str, float]  # FamName -> K at that energy, current held

    def _set(self, simulator, amps):
        super()._set(simulator, amps * self.ev_per_amp)

    def _get(self, simulator):
        return super()._get(simulator) / self.ev_per_amp

    def _after_write(self, ring, energy):
        for name, k_reference in self.reference_strengths.items():
            index = ring.get_uint32_index(name)[0]
            ring[index].K = k_reference * self.reference_energy / energy

    def snapshot_targets(self, simulator):
        return [
            *super().snapshot_targets(simulator),
            *(
                (simulator.element_index(name), "K")
                for name in self.reference_strengths
            ),
        ]
```

## 4D and 6D rings

`solve_orbit` follows the ring: a lattice without an enabled cavity is solved
with `find_orbit4`, one with a cavity enabled (`ring.enable_6d(at.RFCavity)`)
with `find_orbit6`, so an RF frequency change produces the dispersive orbit it
should. In both cases the stability guard is the same: every eigenvalue of the
`find_m66` one-turn matrix must satisfy `|eigenvalue| ≤ 1 + 1e-6`, in all
three planes.

**Rings with radiation enabled are out of scope.** The guards still run — a
damped map passes them — but nothing in this package accounts for the energy
loss per turn, and the orbit such a ring solves to is not the one its
variables describe. Enable the cavity only, and leave radiation off.

## API

| Name | What it is |
|------|-----------|
| `PyATSimulator` | Owns one `at.Lattice`; `solve()`, `element()`, `element_index()`, `unique_element_index()`, `lattice`, `last_solution`, and `snapshot_solution()`/`restore_solution()` for rolling a solve back |
| `LUMEPyATModel` | `ActionModel` over a simulator; atomic `set()`, cached `get()`, batched `reset()` |
| `PyATWritableScalarVariable` | Binds a name to one or more element attributes, each with a weight; the extension point |
| `ElementBinding` | One `(element_name, attribute, index, weight)` entry of a writable's `bindings` |
| `PyATLatticeScalarVariable` | The ring energy as a variable: `ring.energy` plus every cavity's `Energy`, with an `_after_write` hook and `snapshot_targets` for subclasses |
| `PyATReadOnlyScalarVariable` | One transverse coordinate of a monitor's reading |
| `solve_orbit`, `monitor_xy` | Guarded closed-orbit solve (4D or 6D, following the ring) and its monitor readout |
| `apply_misalignment` | dx/dy/roll element misalignment, ported from pySC |
| `OrbitSolveError`, `UnknownElementError`, `AmbiguousElementError` | The exception contract |

### What raises what

Everything that can be settled when a variable is *declared* is settled then,
so the write path raises only about the physics.

| Exception | Raised when |
|-----------|-------------|
| `UnknownElementError` | A name does not reach an element of the lattice — a variable binding, a lookup, or a misalignment key |
| `AmbiguousElementError` | A name reaches more than one, so it addresses neither: two monitors sharing a `FamName` (at `PyATSimulator`) or a variable binding a repeated name (at `LUMEPyATModel`). A subclass of `UnknownElementError` |
| `AttributeError` | A variable declares an attribute its element does not have — in any of its bindings, or in a lattice variable's `snapshot_targets` — caught when `LUMEPyATModel` adopts the variable |
| `IndexError` | A binding declares an `index` past the end of its attribute's sequence, caught when `LUMEPyATModel` adopts the variable |
| `TypeError` | A variable binds neither a lattice element nor the lattice — it is not one of this package's action variables — caught when `LUMEPyATModel` adopts it |
| `ValueError` | A writable variable is declared without the `default_value` that `reset()` writes back to the lattice; a binding's `weight` is zero or not finite; or both `bindings=` and the single-element form are given. Pydantic surfaces each as a `ValidationError`, which is a `ValueError` |
| `ReadOnlyError` | `set()` is called on a read-only variable — an orbit reading is solved for, never written — or one is declared with `read_only=False`. lume-base raises it; it is a subclass of `TypeError` |
| `OrbitSolveError` | A solve cannot be trusted: a non-finite one-turn matrix, a one-turn eigenvalue with `\|eigenvalue\| > 1 + 1e-6` in any plane (the message names it), or a non-finite closed orbit |

Duplicate names are only a problem for names you address. A lattice whose
drifts all share one name — most of them — is fine.

## Development

```bash
uv sync --extra dev
uv run pytest lume_pyat/tests
uv run pre-commit run --all-files
```

## License

Apache-2.0. See [LICENSE](LICENSE).
