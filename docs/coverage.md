# Coverage and applicability

This repository separates completed numerical evidence from potentially meaningful comparisons. **N/A means no equivalent interface or matched native interaction is used in this harness. It is not a passing test, and does not assert that the project could never support such a comparison.** “Not run” means a meaningful comparison has not been included. A library's agreement on F001 is not evidence for its behavior on another row.

## Five-implementation map

| Finding | TorchSim | ASE | LAMMPS | JAX MD | TorchMD |
|---|---|---|---|---|---|
| F001 — default direct-force accumulation | Affected full-list result; half-list and energy-derived controls | Native LJ forces, executed | Native `lj/cut` forces, executed | Native LJ gradient, executed; original soft sphere also executed | Native LJ forces, executed |
| F002 — per-atom energy allocation | Compare sum of `energies` with `energy` | Native LJ per-atom sum control, executed | Native per-atom tally possible; not run for this finding | Native per-particle energy sum, executed and retained in force evidence | N/A: included force interface does not provide the matching per-atom partition |
| F003 — fractional soft-sphere derivative | Species cutoff smaller than global neighbor cutoff; require finite zero force outside species range | N/A: matched native soft sphere not included | N/A: no equivalent native potential configured; similarly named soft interactions are insufficient | Matched native fractional soft-sphere gradient, executed | N/A: matched native soft sphere not included |
| F004 — per-system timestep | Batched API compared with independently stepped systems | Separate native scalar-timestep `VelocityVerlet` runs, executed | N/A to the TorchSim batched argument contract; no independent trajectory included | N/A to the TorchSim batched argument contract; no independent trajectory included | N/A to the TorchSim batched argument contract; no independent trajectory included |
| F005 — unequal-mass initialization | Mean-momentum subtraction; zero total momentum but different convention | Native `Stationary(..., preserve_temperature=False)` common-velocity convention, executed | Not run; no initialization-convention claim | Native initialization executed; also subtracts mean momentum: counterexample to unanimity | Not run; no initialization-convention claim |
| F006 — custom mass export | ASE import/export round trip loses explicit isotope masses | Native mass-preserving trajectory round trip, executed | N/A: not the same conversion endpoint | N/A: not the same conversion endpoint | N/A: not the same conversion endpoint |
| F007 — list conversion dtype | List of `SimState` inputs versus single-input requested conversion | N/A: TorchSim-specific tensor and state API | N/A: TorchSim-specific tensor and state API | N/A: TorchSim-specific tensor and state API | N/A: TorchSim-specific tensor and state API |
| F008 — rejected MC extras | Compare returned observable with fresh model evaluation after rejection | N/A: no matching model-extras acceptance API included | N/A: no matching model-extras acceptance API included | N/A: no matching model-extras acceptance API included | N/A: no matching model-extras acceptance API included |

F001 is the completed five-implementation force comparison, comprising TorchSim plus four external native references. F002–F008 have also been rerun; applicable native controls are marked above. Their current executed results and status are recorded in [findings.json](../findings.json) and the [individual supplemental reports](findings/). They must not be described collectively as seven more five-engine validations.

## What each finding can support

| ID | Classification | Bounded conclusion |
|---|---|---|
| F001 | Force accumulation defect | The default full-list direct-force wrapper doubles the tested symmetric atomic forces; existing controls agree with four external implementations. |
| F002 | Output allocation inconsistency | Optional per-atom energies fail to sum to the same model's total energy in the reproduced pair case. Atomic energy partitions can vary by convention; failure of this stated sum is the invariant. |
| F003 | Nonfinite derivative defect | A masked fractional power can have a zero forward energy yet produce nonfinite differentiated forces outside a species-specific interaction range. The included test concerns the reproduced multi-species configuration. |
| F004 | Batched argument handling defect | A three-element per-system timestep can broadcast onto coordinates instead of selecting each atom's system timestep. It does not show failure of scalar timesteps. |
| F005 | Qualified initialization convention discrepancy | Mean-momentum and common-velocity projections differ for unequal masses. Both remove total momentum. Stronger distributional claims require an explicit intended ensemble or contract. |
| F006 | IO data preservation defect | Explicit isotope/custom masses do not survive the tested TorchSim-to-ASE export, even though import retained them. |
| F007 | API conversion inconsistency | Requested float64 conversion is ignored for a list of float32 states, unlike a single state. CPU dtype is reproduced; accelerator device behavior is not inferred from an unperformed run. |
| F008 | Returned-state observable inconsistency | A rejected swap restores positions and energy while retaining a model extra evaluated on the rejected proposal. A synthetic deterministic model makes the mismatch observable. |

For F002, ASE's shifted LJ totals may differ from unshifted LJ totals; its internal per-atom sum is still a valid allocation control. LAMMPS documents a native [`compute pe/atom`](https://docs.lammps.org/compute_pe_atom.html), but that supplemental comparison has not been run here. For F003, the native JAX MD overlap soft-sphere law is the matched potential; a generic inverse-power “soft sphere” or similarly named soft interaction would not be equivalent. For F004, independent scalar runs use native ASE [`VelocityVerlet`](https://docs.ase-lib.org/ase/md.html#ase.md.verlet.VelocityVerlet) to test the numerical expectation without inventing a corresponding batched API in other engines. F006 uses ASE's native [trajectory IO](https://docs.ase-lib.org/ase/io/trajectory.html) as its mass-preservation control.

For F005, `P=sum(p_i)` and `M=sum(m_i)`. TorchSim uses `p_i'=p_i-P/N`; the common-velocity convention uses `p_i'=p_i-m_i*P/M`. Each has `sum(p_i')=0`. Only the latter subtracts the same velocity from every atom. ASE's [native source](https://docs.ase-lib.org/_modules/ase/md/velocitydistribution.html#Stationary) implements that common-velocity subtraction. Native JAX MD 0.2.29 initialization was also executed and implements the former. The comparison is qualified accordingly.

The F005 ensemble check is a separate conditional inference. With masses `[1,4]` and `kT=1`, the mean-momentum projection predicts a component variance of `1.25` for both atoms, while conditioning the original independent Gaussian momenta on total momentum zero predicts `0.8`. The measured TorchSim variances over 32,768 independent two-atom systems are approximately `1.251`. This demonstrates a distribution difference under that intended-ensemble assumption; it does not turn agreement between TorchSim and JAX MD into independent bug corroboration.

## Unexamined dimensions

No exhaustive potential, material, particle-count, GPU, periodic-image, long-trajectory, thermostat, stress, or scientific-publication survey was performed. Source inheritance broadens the identified exposure of F001 without numerically validating every derived workflow. The registry deliberately keeps source exposure, observed output, external corroboration, and semantic uncertainty separate.
