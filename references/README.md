# Native computation adapters

These are portable adaptations of the original executed audit scripts. They
call the installed native engines and write raw numeric results; no adapter
implements an expected force formula or imports TorchSim as a reference.

| Adapter | Native computation | Pinned source/version |
| --- | --- | --- |
| `native_torchsim.py` | `PairForcesModel` full/half lists and `PairPotentialModel`, default and `torch_nl_n2` neighbors | [TorchSim d6d8e9fc30129e01a05b551c2d71912821c9b68b](https://github.com/torchsim/torch-sim/tree/d6d8e9fc30129e01a05b551c2d71912821c9b68b) |
| `native_ase.py` | `ase.calculators.lj.LennardJones`, `Atoms.get_forces` | [ASE 3.29.0](https://pypi.org/project/ase/3.29.0/) |
| `native_lammps.py` | Actual shared library, built-in `pair_style lj/cut`, `gather_atoms` | [LAMMPS 2025.7.22.4.0](https://pypi.org/project/lammps/2025.7.22.4.0/) with MPICH 5.0.2 |
| `native_jaxmd.py` | `energy.lennard_jones_pair`, `quantity.force`, `space.free`; native soft-sphere control | [JAX MD 0.2.29](https://pypi.org/project/jax-md/0.2.29/), JAX/JAXlib 0.12.0.1 |
| `native_torchmd.py` | `YamlForcefield` → `Parameters` → `Forces.compute` | [TorchMD 09484183e34af78bb69b27a0149aeccf1318dbc7](https://github.com/torchmd/torchmd/tree/09484183e34af78bb69b27a0149aeccf1318dbc7), not TorchMD-Net |

All force vectors use atom 0 at the origin, atom 1 at positive x, no periodic
interactions, CPU float64, and reduced force `F*sigma/epsilon`. With epsilon=1 eV
and sigma=1 angstrom the numbers are also eV/angstrom. TorchMD emits both native
epsilon=1 kcal/mol and epsilon=1 eV runs; the latter conversion is independently
checked by the consumer. Its conversion uses `scipy.constants`, not a force law.

ASE uses its native nonsmooth energy shift at cutoff. LAMMPS, JAX MD below its
switch onset, and TorchMD return unshifted energies. Therefore the force runner
does not compare energy values across engines. Total and per-particle native
energy fields from ASE, LAMMPS, and JAX MD are retained for separate checks.
The soft-sphere alpha=2, r=0.5 control is native only in TorchSim and JAX MD;
no equivalent calculation is claimed for the other adapters.

The original scripts' SHA-256 hashes are in `provenance.json`. Adaptations remove
machine-specific paths, accept an output filename, verify installed source
provenance where required, and capture native energies. Runtime logs, temporary
force-field files, caches, virtual environments, and downloaded sources are not
evidence and are not tracked.
