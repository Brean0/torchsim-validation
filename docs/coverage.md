# Coverage and applicability

This page records the completed numerical evidence and the comparisons that remain open. **N/A means this harness uses no equivalent interface or matched native interaction. It is not a passing test or a claim that the project could never support the comparison.** “Not run” means a meaningful comparison has not been included. Agreement on F001 provides evidence for that finding alone.

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

F001 is the completed five-implementation force comparison: TorchSim plus four external native references. F002–F008 have also been rerun, with applicable native controls marked above. Their current executed results and status are recorded in [findings.json](../findings.json) and the [individual supplemental reports](findings/). These seven findings are not seven more five-engine validations.

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

For F002, ASE's shifted LJ totals may differ from unshifted LJ totals. Its internal per-atom sum remains a valid allocation control. LAMMPS documents a native [`compute pe/atom`](https://docs.lammps.org/compute_pe_atom.html), but that supplemental comparison has not been run here. For F003, the matched potential is the native JAX MD overlap soft-sphere law; a generic inverse-power “soft sphere” or similarly named soft interaction would not be equivalent. For F004, independent scalar runs use native ASE [`VelocityVerlet`](https://docs.ase-lib.org/ase/md.html#ase.md.verlet.VelocityVerlet) to test the numerical expectation. This does not imply a corresponding batched API in other engines. F006 uses ASE's native [trajectory IO](https://docs.ase-lib.org/ase/io/trajectory.html) as its mass-preservation control.

For F005, `P=sum(p_i)` and `M=sum(m_i)`. TorchSim uses `p_i'=p_i-P/N`; the common-velocity convention uses `p_i'=p_i-m_i*P/M`. Each has `sum(p_i')=0`. Only the latter subtracts the same velocity from every atom. ASE's [native source](https://docs.ase-lib.org/_modules/ase/md/velocitydistribution.html#Stationary) implements that common-velocity subtraction. The executed native JAX MD 0.2.29 initialization uses the same convention as TorchSim. F005 remains a qualified comparison.

The F005 ensemble check is a separate conditional inference. With masses `[1,4]` and `kT=1`, the mean-momentum projection predicts a component variance of `1.25` for both atoms, while conditioning the original independent Gaussian momenta on total momentum zero predicts `0.8`. The measured TorchSim variances over 32,768 independent two-atom systems are approximately `1.251`. This shows a distribution difference under that intended-ensemble assumption. Agreement between TorchSim and JAX MD remains agreement on the convention, not independent bug corroboration.

## Unexamined dimensions

The work does not include an exhaustive survey of potentials, materials, particle counts, GPUs, periodic images, long trajectories, thermostats, stress, or scientific publications. Source inheritance broadens the identified exposure of F001; numerical validation of every derived workflow remains outside the evidence. The registry records source exposure, observed output, external corroboration, and semantic uncertainty separately.
