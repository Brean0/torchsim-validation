# TorchSim validation

This repo documents selected TorchSim findings and provides the code, inputs and results needed to check them. Working controls are included so each discrepancy can be traced to a specific implementation or option.

Audited TorchSim commit: [`d6d8e9fc30129e01a05b551c2d71912821c9b68b`](https://github.com/TorchSim/torch-sim/commit/d6d8e9fc30129e01a05b551c2d71912821c9b68b). It was still upstream `main` when checked on November 6, 2026. The repo contains validation evidence and separately prepared local fix proposals; the audited upstream implementation and upstream submission history are unchanged.

## Findings with native reference controls

These five findings have native reference comparisons supporting the reported discrepancy. Each links to its own report with the setup, observed result, source location and reproduction commands. The evidence differs by case; F001 is the completed five-library comparison.

| ID | Finding | Native reference evidence |
|---|---|---|
| [F001](docs/findings/F001.md) | Default direct-force accumulation doubles symmetric pair forces | ASE, LAMMPS, JAX MD and TorchMD agree; TorchSim's energy-derived and half-list controls also agree |
| [F002](docs/findings/F002.md) | Per-atom energies sum to twice the reported total | Native ASE energy-allocation control; JAX MD per-particle data retained |
| [F003](docs/findings/F003.md) | Fractional soft-sphere differentiation produces nonfinite forces outside a species cutoff | Native JAX MD returns finite zero forces for the matched configuration |
| [F004](docs/findings/F004.md) | Per-system timesteps broadcast onto coordinate axes | Independent native ASE scalar-timestep runs provide the displacement control |
| [F006](docs/findings/F006.md) | ASE export drops explicit isotope or custom masses | A native ASE mass-preserving round trip retains the input masses |

## Other reproduced observations

These cases use state or API consistency checks, or require further interpretation of the intended behavior.

| ID | Finding | Evidence and status |
|---|---|---|
| [F007](docs/findings/F007.md) | State-list initialization ignores the requested dtype conversion | Reproduced CPU API inconsistency: the same request works for a single state |
| [F008](docs/findings/F008.md) | A rejected Monte Carlo swap retains properties from the rejected proposal | Reproduced returned-state inconsistency, checked against a fresh model evaluation |
| [F005](docs/findings/F005.md) | Unequal-mass initialization differs from common-velocity drift removal | Qualified: ASE uses a different projection, while JAX MD shares TorchSim's rule; both remove total momentum |

The initial audit produced eight reproducible findings. They are a bounded inventory, with the evidence and limits tracked in [findings.json](findings.json) and the [coverage matrix](docs/coverage.md). A missing equivalent feature in another library is marked **not applicable**.

For F001, there is **one independently corroborated force-accumulation root cause**, exposed through at least two named public classes: `PairForcesModel` and its `ParticleLifeModel` subclass. Those entry points count as one defect. The total number of defects in TorchSim, affected users and effects on published research remain unknown.

The [source and release history](docs/history.md) records how long the affected implementations have existed, with historical-runtime limits stated separately. Each report also includes a proposed change.

The [local PR review package](pull-requests/README.md) contains five separate proposed fixes, regression tests and draft descriptions. The [one-page overview](output/pdf/torchsim-validation-overview.pdf) summarizes the findings without code or setup instructions.

## Run the checks

Check the saved force-comparison evidence without installing the simulation engines:

```sh
python3 scripts/run_force_comparison.py
```

Check the saved supplemental evidence:

```sh
python3 -m reproductions.validate evidence/other-findings/results.json \
  --jax-reference evidence/other-findings/jaxmd.json
```

Exit zero means the checker verified the stated observations and controls. An upstream fix may change that result. Saved evidence and fresh engine executions are labeled separately.

Install the pinned native engines into separate local environments and recompute the force comparison:

```sh
python3 scripts/run_force_comparison.py --setup --live
```

Setup requires Python 3.12, [uv](https://docs.astral.sh/uv/), network access and compatible CPU packages. Native executions have been verified on macOS arm64. Other platforms remain unverified. Use the interpreter options shown by `--help` to supply existing environments.

```sh
python3 scripts/run_force_comparison.py --help
uv run --with pytest==9.2.0 python -m pytest tests -q
```

Recompute the supplemental cases using the appropriate isolated environments:

```sh
.venvs/torchsim/bin/python -m reproductions.run_other_findings
.venvs/jaxmd/bin/python -m reproductions.native_jaxmd
```

The [verification record](docs/verification.md) lists the checks performed, setup limits and independent review results. Individual reports under [docs/findings](docs/findings) provide case-specific commands.

## How the comparisons work

An interaction model supplies an energy or force law. The simulation library handles the surrounding operations, including neighbor lists, force accumulation, integration and state conversion. [The force-path explanation](docs/force-paths.md) shows how a correct force helper can still produce an incorrect final atomic force.

The ASE, LAMMPS, JAX MD and TorchMD probes call their native implementations. They neither import TorchSim nor implement its force formula. Units, cutoff switches and energy shifts are matched or accounted for before results are compared. [The methodology](docs/methodology.md) explains those choices and the conclusions each type of evidence supports.

Future upstream PRs should address demonstrated root causes and include the relevant reproduction, version and source references. The qualified initialization finding needs its intended distribution stated before a universal defect is claimed.

## Files

- `references/`: native engine probes and TorchSim controls.
- `scripts/`: setup, execution and comparison tooling.
- `requirements/`: pinned inputs for isolated environments.
- `evidence/`: numerical results and provenance, with machine paths removed.
- `reproductions/`: supplemental cases.
- `tests/`: checks that the evidence consumer rejects invalid or contradictory data.
- `docs/`: methodology, coverage and individual finding reports.
