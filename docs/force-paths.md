# One accumulation defect across several routes

TorchSim provides separate energy-derived and direct-force model paths. The default direct-force path is affected by F001; the matched energy-derived force calculation is a control. This distinction is visible in the [pinned module](https://github.com/TorchSim/torch-sim/blob/d6d8e9fc30129e01a05b551c2d71912821c9b68b/torch_sim/models/pair_potential.py).

## How pair forces become atomic forces

The direct-force callable returns a scalar radial force for each listed pair. The wrapper converts it to a vector using the pair direction, then adds contributions to the atoms' force arrays. The scalar helper can return the correct value while the wrapper produces an incorrect final atomic force.

For two atoms, a full neighbor list contains both directed entries `(i,j)` and `(j,i)`. For a symmetric interaction, each already covers the corresponding source atom's contribution. Accumulating a Newton pair on **both endpoints for both entries** counts each contribution twice. A half list keeps one unordered pair and therefore requires the two endpoint additions exactly once.

The table shows how each TorchSim route handles the list:

| Path at the audited commit | List and accumulation | Observed matched force |
|---|---|---|
| `PairForcesModel()` | Default full list; adds each row to both endpoints | Twice the native references |
| `PairForcesModel(reduce_to_half_list=True)` | One unordered pair; adds to both endpoints | Agrees for the tested symmetric forces |
| `PairPotentialModel()` | Full list; differentiates pair energy and accumulates to the source atom | Agrees |
| `PairPotentialModel(reduce_to_half_list=True)` | One unordered pair with two endpoint updates | Source contains this separate branch; not needed for the reported full-list control |

The relevant source is [direct-force accumulation, lines 580–588](https://github.com/TorchSim/torch-sim/blob/d6d8e9fc30129e01a05b551c2d71912821c9b68b/torch_sim/models/pair_potential.py#L580-L588), compared with [energy-derived force accumulation, lines 431–441](https://github.com/TorchSim/torch-sim/blob/d6d8e9fc30129e01a05b551c2d71912821c9b68b/torch_sim/models/pair_potential.py#L431-L441). The existing half-list option is a diagnostic control for symmetric interactions, not a source patch or a general workaround for asymmetric interactions.

## Which named classes use which route?

At this commit, source inspection identifies two named public entry classes on the direct-force route: `PairForcesModel` itself and its `ParticleLifeModel` subclass. [`ParticleLifeModel`](https://github.com/TorchSim/torch-sim/blob/d6d8e9fc30129e01a05b551c2d71912821c9b68b/torch_sim/models/particle_life.py#L57-L121) explicitly calls the parent with `reduce_to_half_list=False`. The source establishes its exposure to the same accumulation code. A separate five-engine ParticleLife comparison remains unperformed. Its supplied helper ignores the species arguments and uses the bound scalar parameters, while the general `PairForcesModel` API also permits genuinely direction-dependent callables.

[`SoftSphereModel` and `SoftSphereMultiModel`](https://github.com/TorchSim/torch-sim/blob/d6d8e9fc30129e01a05b551c2d71912821c9b68b/torch_sim/models/soft_sphere.py#L175-L378) inherit `PairPotentialModel`. The original force-doubling reproduction supplies `soft_sphere_pair_force` to `PairForcesModel`; it does not show doubled forces from the energy-derived `SoftSphereModel`. The soft-sphere per-atom energy and fractional-derivative findings concern different behavior and have separate evidence.

Asymmetric directed interactions require separate analysis. The current two-endpoint accumulation mixes their directed force laws, so an exact factor of two is not established for every such result. The exact factor of two is established for the symmetric LJ and soft-sphere cases here. Turning a directed interaction into a half list can discard a physically distinct directed interaction, so that control cannot be generalized indiscriminately.

## Counting

Count F001 as **one accumulation root cause**. Its two named entry classes, two tested interaction laws, three LJ distances, and two neighbor backends describe exposure and coverage. They are not additional bugs. User-defined wrappers or subclasses could also use that code, but their number and behavior have not been inventoried. Neither this path analysis nor the eight initial regressions establishes the defect count for the whole codebase.
