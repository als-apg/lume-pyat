# lume-pyat

pyAT-specific implementation of [LUME](https://github.com/lume-science) model
classes for virtual accelerators.

`lume-pyat` sits between [lume-base](https://github.com/lume-science/lume-base)
and [pyAT](https://github.com/atcollab/at): it wraps a lattice you build
yourself in a `PyATSimulator`, binds free-form variable names to element
attributes in native pyAT units, and exposes the whole thing as a lume-base
`ActionModel`. It is a sibling of
[lume-cheetah](https://github.com/lume-science/lume-cheetah) and follows the
same conventions.

The package is deliberately facility-agnostic. It ships no lattice data and
knows nothing about any particular machine's naming, units, or calibration —
mapping control-system addresses and engineering units onto simulator
parameters belongs to the layer above.

## Installation

```bash
pip install lume-pyat
```

## Status

Early development. A usage example and the full API documentation land with
the first release.

## License

Apache-2.0. See [LICENSE](LICENSE).
