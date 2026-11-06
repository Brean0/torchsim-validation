# What this validation establishes

The strongest completed comparison concerns **F001, direct pair-force accumulation**. Native ASE, LAMMPS, JAX MD, and TorchMD calculations agree for three two-atom Lennard-Jones configurations. TorchSim's default full-list `PairForcesModel` returns twice those forces. Two existing TorchSim paths—the half-list option for a symmetric force and the energy-derived `PairPotentialModel`—agree with the references. This is evidence about specific paths at commit [`d6d8e9f`](https://github.com/TorchSim/torch-sim/tree/d6d8e9fc30129e01a05b551c2d71912821c9b68b), not every force calculation in TorchSim.

The eight registry entries originate from eight initial failing regressions and have been rerun with retained numerical evidence. Their evidence levels differ. Seven concern output, numerical, or API consistency; F005 is a qualified discrepancy between initialization conventions. Eight is neither an estimate of defects across the codebase nor the number of independently cross-validated physics bugs. See [coverage](coverage.md) and the machine-readable [registry](../findings.json).

## Models, engines, and simulation procedures

A **model** supplies an interaction law and outputs such as energy and atomic forces. A simulation library supplies machinery around it: neighbor lists, state handling, integration, random initialization, conversion, and Monte Carlo acceptance. Here “five engines” is shorthand for five independently installed implementations or libraries: TorchSim, ASE, LAMMPS, JAX MD, and TorchMD. TorchMD means the molecular simulation project, not TorchMD-Net.

Matching a potential name alone is insufficient. Parameters, length and energy units, boundary conditions, cutoff rules, precision, and the quantity being compared must agree. Energy and force tests also differ from tests of **simulation procedures**, such as a timestep update or a rejected Monte Carlo move. An IO contract or batched-state API often has no matching interface in another engine; an invariant is then more informative than an artificial five-way comparison.

## Equivalent force cases

The Lennard-Jones comparison uses two atoms at `(0,0,0)` and `(r,0,0)`, with `r/sigma` equal to 1.1, 1.3, and 1.5. All implementations use native 12–6 LJ, `epsilon=sigma=1`, cutoff 2.5, no periodic interactions, CPU execution, and float64. The set includes repulsive and attractive forces; it does not rely on a zero at equilibrium. The external scripts call each project's native implementation and do not import TorchSim or duplicate its force expression.

Forces are normalized as `F*sigma/epsilon`. ASE, LAMMPS, and TorchSim use eV and angstrom inputs. TorchMD uses its native kcal/mol convention, with normalization and a separate physical `epsilon=1 eV` conversion check. JAX MD uses reduced parameters and `space.free()`.

JAX MD's native switch starts at 2.0, above every tested separation, so it is identically one here. ASE's constant energy shift at 2.5 changes total energies but leaves these forces unchanged. LAMMPS and TorchMD use unshifted LJ. Consequently, raw total energies must not be compared across these configurations without accounting for their offsets. Relevant primary sources are [LAMMPS LJ](https://docs.lammps.org/pair_lj.html), [ASE LJ source](https://docs.ase-lib.org/_modules/ase/calculators/lj.html), [JAX MD energy API](https://jax-md.readthedocs.io/en/main/jax_md.energy.html), and [pinned TorchMD forces](https://github.com/torchmd/torchmd/blob/09484183e34af78bb69b27a0149aeccf1318dbc7/torchmd/forces.py).

The complete two-by-three force arrays are compared with `rtol=1e-12` and `atol=5e-12`. The four external implementations' largest disagreement in the retained original run is approximately `1.55e-15` in normalized units. The existing half-list and energy-derived controls are checked independently. The default neighbor dispatcher and explicit `torch_nl_n2` backend give the same TorchSim discrepancy. This isolates accumulation after neighbor construction more strongly than a single-backend result would.

The original soft-sphere case supplies a second interaction-law control: native JAX MD gives forces `(-0.5,0,0)` and `(+0.5,0,0)` at `r=0.5`, `alpha=2`, and `epsilon=sigma=1`; TorchSim's default direct-force wrapper gives twice those values. This is not another root cause or a second five-engine comparison.

## Evidence and reproducibility

- **Native numerical comparison:** independently installed project code produces recorded outputs for equivalent inputs. This is the strongest evidence used for F001.
- **Invariant or API consistency check:** compare outputs that must agree within the stated contract, such as the sum of per-atom energies and total energy, or a stored observable and a fresh model evaluation.
- **Source inspection:** establishes where a behavior occurs and whether a class inherits the affected path. It does not substitute for an unperformed numerical test.
- **Qualified convention discrepancy:** reproduces a difference while retaining an unresolved question about intended semantics. F005 belongs here.

Saved JSON enables lightweight validation of the observations. A live run executes the installed implementations again. Passing the evidence validator means the declared discrepancy and controls were observed; it does not mean the audited implementation is fixed. A future fix may intentionally make a validator that expects the old discrepancy fail.

The force evidence retains versions, settings, vectors, and package/source identity. The TorchSim and TorchMD source URLs are commit-pinned. JAX MD is pinned by [release](https://pypi.org/project/jax-md/0.2.29/) and wheel digest, including an installed-source digest in its evidence. ASE and LAMMPS package versions are pinned by the harness requirements. Online documentation links explain APIs and may change; they are not a substitute for recorded package identities.

## Limits

These small deterministic cases expose particular behavior. They do not certify all potentials, materials, GPU kernels, periodic geometries, dynamics, or sampling workflows. No downstream scientific result, user harm, or published calculation has been assessed. A correct control for F001 does not certify other outputs from that class: `PairPotentialModel`'s forces can pass while its optional per-atom energy allocation fails F002.

F005 deserves special care. TorchSim subtracts mean momentum, which **does make total momentum and center-of-mass velocity zero** for the reproduced multi-atom system. ASE's usual common-velocity subtraction instead removes `m_i*P/M` from each momentum and preserves relative velocities. These differ for unequal masses. The pinned native JAX MD initialization was also run and subtracts mean momentum. The supplemental ensemble check quantifies the resulting variance difference under an explicit conditioned-canonical-Gaussian assumption. Thus the reproduced difference supports a qualified discussion of initialization semantics; it does not establish failure to remove COM motion or unanimous disagreement among other libraries.
