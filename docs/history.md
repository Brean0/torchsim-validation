# How long the affected code has existed

Checked on November 6, 2026 against the full upstream Git history available in the audit checkout. The audited current implementation is [`d6d8e9fc30129e01a05b551c2d71912821c9b68b`](https://github.com/TorchSim/torch-sim/commit/d6d8e9fc30129e01a05b551c2d71912821c9b68b).

These dates describe source history, not the number of affected users or calculations. Current-version reproductions establish the observed discrepancies. Historical source inspection and tag ancestry establish when the relevant implementation was present; the audit has not executed every historical release.

## Five findings with native reference controls

| Finding | Earliest confirmed affected implementation | Source age on November 6, 2026 | First GitHub release containing that implementation |
|---|---|---|---|
| [F001: direct-force accumulation](findings/F001.md) | [March 2, 2026, general pair models](https://github.com/TorchSim/torch-sim/commit/e8bc7a3b7c15906464721e4e8589e6b81fadc446) | 249 days | [v0.6.0, April 24, 2026](https://github.com/TorchSim/torch-sim/releases/tag/v0.6.0) |
| [F002: per-atom energy allocation](findings/F002.md) | [March 2, 2026, general pair models](https://github.com/TorchSim/torch-sim/commit/e8bc7a3b7c15906464721e4e8589e6b81fadc446) | 249 days | v0.6.0, April 24, 2026 |
| [F003: fractional soft-sphere differentiation](findings/F003.md) | [March 2, 2026, scalar and species-dependent helpers](https://github.com/TorchSim/torch-sim/commit/e8bc7a3b7c15906464721e4e8589e6b81fadc446) | 249 days | v0.6.0, April 24, 2026 |
| [F004: per-system timestep broadcasting](findings/F004.md) | [March 24, 2025, vector-timestep API documented alongside direct multiplication](https://github.com/TorchSim/torch-sim/commit/1424be591a32783b3b33c284f0eef059d4f33321) | At least 592 days | [v0.1.0, April 3, 2025](https://github.com/TorchSim/torch-sim/releases/tag/v0.1.0) |
| [F006: ASE export drops custom masses](findings/F006.md) | [March 3, 2025, initial high-level state exporter](https://github.com/TorchSim/torch-sim/commit/fac94afcc639fd0f4661ebe55a906c804c4246be) | 613 days | v0.1.0, April 3, 2025 |

The first three have therefore been in a published GitHub release for **196 days**. The two older implementations have been in GitHub releases for **582 days**. These are calendar-day differences; they do not imply that all configurations are affected.

F001, F002 and F003 are visible in the original general-pair implementation, before the [March 5 helper reorganization](https://github.com/TorchSim/torch-sim/commit/84986c1830c63fad48e740d4e994c4069e48d08d). The reorganization moved the code and changed accumulation syntax without removing these mechanisms. For F004, the underlying multiplication appears in the initial March 3 batched code, but March 24 is the conservative date where the per-system vector contract is explicit. F006 is visible before the exporter moved from `torchsim/runners.py` to `torch_sim/io.py`; the later move is not its introduction date.

## Other reproduced observations

| Observation | Earliest confirmed source snapshot | Source age | First containing release |
|---|---|---|---|
| [F007: requested dtype ignored for state lists](findings/F007.md) | [October 9, 2025, list-state initialization branch](https://github.com/TorchSim/torch-sim/commit/9151970) | At least 393 days | v0.4.0, October 9, 2025 |
| [F008: rejected proposal extras retained](findings/F008.md) | [April 9, 2026, extra-property scheme](https://github.com/TorchSim/torch-sim/commit/8e920141fd4235a36f1fd79dda782f4612a18773) | 211 days | v0.6.0, April 24, 2026 |
| [F005: unequal-mass initialization convention](findings/F005.md) | [March 3, 2025, mean-momentum subtraction](https://github.com/TorchSim/torch-sim/commit/fac94afcc639fd0f4661ebe55a906c804c4246be) | At least 613 days | v0.1.0, April 3, 2025 |

F005 remains qualified. Its age describes a numerical convention, not an established period of incorrect behavior: native JAX MD shares the same projection, while ASE uses another one.

## How the dates were checked

The audit followed function moves through Git history, inspected the older function bodies, and checked the earliest release tags containing those commits. GitHub release publication dates were read separately; a source commit date is not a release date. Package-index and archival publication dates were not used and can differ from GitHub release dates. For example:

```sh
git log --reverse -S 'def state_to_atoms' -- '*.py'
git show e8bc7a3b:torch_sim/models/pair_potential.py
git show fac94af:torchsim/runners.py
git show 1424be5:torch_sim/integrators.py
git tag --contains e8bc7a3b --sort=version:refname
gh api repos/TorchSim/torch-sim/releases --jq '.[] | {tag_name,published_at}'
```

The relevant mechanisms remain in the audited current source. The audit does not establish effects on published research, how often the affected options were used, or whether private workarounds exist.
