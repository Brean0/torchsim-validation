# TorchSim validation

Reproducible numerical comparisons and state-consistency checks for selected TorchSim findings. The purpose is to make the evidence easy to inspect and rerun, including controls that work correctly.

Audited TorchSim commit: [`d6d8e9fc30129e01a05b551c2d71912821c9b68b`](https://github.com/TorchSim/torch-sim/commit/d6d8e9fc30129e01a05b551c2d71912821c9b68b). This was still upstream `main` when checked on November 6, 2026. No upstream implementation changes or submissions are included.

## Start here

Check the saved force-comparison evidence without installing the simulation engines:

```sh
python3 scripts/run_force_comparison.py
```

Install the pinned native engines into separate local environments and recompute:

```sh
python3 scripts/run_force_comparison.py --setup --live
```

Setup requires Python 3.12, [uv](https://docs.astral.sh/uv/), network access, and compatible CPU packages for your platform. Native executions have been verified on macOS arm64. Other platforms must be validated rather than assumed equivalent. Existing environments can be supplied using the interpreter options shown by `--help`.

```sh
python3 scripts/run_force_comparison.py --help
uv run --with pytest==9.2.0 python -m pytest tests -q
```

Exit zero from the evidence checker means the stated observations and controls were verified. It does **not** mean an upstream bug has been fixed. Saved evidence and fresh engine executions are labeled separately.

## First finding: one force path doubles interactions

The same native Lennard-Jones interaction is evaluated by ASE, LAMMPS, JAX MD, TorchMD and TorchSim. Two atoms are placed at `(0, 0, 0)` and `(r, 0, 0)`, with epsilon=sigma=1, cutoff=2.5, no periodic boundaries, and CPU float64.

Force on the first atom along x, in normalized units `F*sigma/epsilon`:

| Engine or TorchSim option | r=1.1 | r=1.3 | r=1.5 |
|---|---:|---:|---:|
| ASE | -1.588095390 | 2.239979930 | 1.158028831 |
| LAMMPS | -1.588095390 | 2.239979930 | 1.158028831 |
| JAX MD | -1.588095390 | 2.239979930 | 1.158028831 |
| TorchMD | -1.588095390 | 2.239979930 | 1.158028831 |
| TorchSim `PairPotentialModel` control | -1.588095390 | 2.239979930 | 1.158028831 |
| TorchSim `PairForcesModel`, half-list control | -1.588095390 | 2.239979930 | 1.158028831 |
| **TorchSim `PairForcesModel`, default full list** | **-3.176190780** | **4.479959860** | **2.316057662** |

The four independent reference engines agree to approximately `1.6e-15`; the affected TorchSim option returns twice their force. Both a repulsive case and attractive cases are included. The second atom's force is opposite, and all y/z components are zero.

Native JAX MD also confirms the original soft-sphere example: at distance 0.5 with epsilon=sigma=1 and alpha=2, its forces are -0.5/+0.5 while the affected TorchSim path gives -1/+1.

### Why the working TorchSim controls matter

An **interaction model** specifies the energy or force law. A **simulation engine** organizes atoms, finds neighbors and accumulates forces. TorchSim supports several paths through that machinery:

- `PairPotentialModel` starts from pair energies and differentiates them to obtain forces. Its tested force accumulation agrees with the four references.
- `PairForcesModel` starts from a supplied pair-force function. Its default full neighbor list contains both A-to-B and B-to-A. Accumulating both endpoints for both entries counts each interaction twice.
- Its existing half-list option keeps one entry per pair and agrees with the references for these symmetric forces.

The result localizes a defect to a specific path, rather than treating every TorchSim model as incorrect. `ParticleLifeModel` inherits the same direct-force implementation; that is another exposed entry point for the same root cause, not an additional independent bug.

Read [the force-path explanation](docs/force-paths.md), [methodology](docs/methodology.md), and [applicability matrix](docs/coverage.md) for the detailed scope and source links.

## Other findings and evidence levels

The initial targeted audit produced eight reproducible findings. Their status is tracked individually in [findings.json](findings.json) and the [coverage matrix](docs/coverage.md).

- Physical calculations are compared with authentic native implementations when the quantities and conventions match.
- Export, dtype and Monte Carlo property handling can instead require round-trip or internal-consistency checks. Those are not presented as five-engine physics comparisons.
- The unequal-mass initialization finding is qualified: TorchSim and JAX MD share mean-momentum subtraction, while ASE's drift removal preserves relative velocities. Zero total momentum alone does not distinguish the methods. The intended initialization distribution must be stated before claiming a universal defect.
- A missing equivalent feature is **not applicable**, never a passing comparison.

Validate the saved supplemental evidence without installing engines:

```sh
python3 -m reproductions.validate evidence/other-findings/results.json \
  --jax-reference evidence/other-findings/jaxmd.json
```

Recompute it using the appropriate isolated environments:

```sh
.venvs/torchsim/bin/python -m reproductions.run_other_findings
.venvs/jaxmd/bin/python -m reproductions.native_jaxmd
```

Supplemental numerical reproductions and their individual commands are documented under [docs/findings](docs/findings).

## What can be counted

There is **one independently corroborated force-accumulation root cause**, exposed through at least two named public classes: `PairForcesModel` and its `ParticleLifeModel` subclass. The initial eight findings are a bounded audit inventory, not an estimate of all defects in TorchSim, and not eight equally corroborated physics errors.

No repository-wide defect total, affected-user count, or impact on published research has been established. Future upstream PRs should be scoped to demonstrated root causes, with reproducible evidence and source/version references.

## Layout

- `references/`: native engine probes and the TorchSim probes/controls.
- `scripts/`: setup, execution and comparison tooling.
- `requirements/`: pinned inputs for isolated environments.
- `evidence/`: sanitized numerical results and provenance.
- `reproductions/`: supplemental cases for the remaining findings.
- `tests/`: checks that the evidence consumer rejects invalid or contradictory data.
- `docs/`: methodology, applicability and finding-specific explanations.

The ASE, LAMMPS, JAX MD and TorchMD reference probes do not import TorchSim or implement its force formula. They invoke the reference engines' own interaction implementations. Cutoff switches, energy shifts and unit conversions are documented explicitly; agreement is claimed only for the tested configurations.

See [verification results and setup limits](docs/verification.md) for the executed checks and independent review record.
