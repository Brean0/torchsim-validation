# Five proposed TorchSim fixes

These are five separate local changes for the findings with native reference controls. Each patch contains its implementation change and regression tests. Each draft links to the corresponding evidence report.

All five branches start from audited upstream commit `d6d8e9fc30129e01a05b551c2d71912821c9b68b`. The original audit checkout and its saved measurements remain unchanged. Upstream PRs have not been opened; this package is ready for Brian to review before publication.

| Finding | Proposed fix | Patch | PR description | Final focused checks | Earlier full-suite checks |
|---|---|---|---|---|---|
| [F001](../docs/findings/F001.md) | Accumulate full-list forces on the source atom; use both endpoints only for half lists | [Patch](../patches/F001.patch) | [Draft](F001.md) | 51 passed | 1009 passed, 208 skipped |
| [F002](../docs/findings/F002.md) | Split each weighted pair energy equally between atoms | [Patch](../patches/F002.patch) | [Draft](F002.md) | 59 passed | 1007 passed, 208 skipped |
| [F003](../docs/findings/F003.md) | Make the inactive fractional-power input safe before differentiation | [Patch](../patches/F003.patch) | [Draft](F003.md) | 44 passed | 1006 passed, 208 skipped |
| [F004](../docs/findings/F004.md) | Map each vector timestep to its system's atoms | [Patch](../patches/F004.patch) | [Draft](F004.md) | 47 passed | 1011 passed, 208 skipped |
| [F006](../docs/findings/F006.md) | Pass explicit masses into ASE export | [Patch](../patches/F006.patch) | [Draft](F006.md) | 121 passed | 1005 passed, 208 skipped |

The full suite ran independently on every branch with no test failures. F002, F003 and F004 then gained additional regression coverage or formatting changes; their production code stayed the same and the final focused suites passed. Each full run emitted three PyTables unclosed-file warnings at interpreter shutdown. Optional and accelerator-specific tests were skipped. The coverage-enabled command could not run in the F001/F002 environment because `pytest-cov` was absent; the plain repository-wide suite passed. All five final changes pass Ruff 0.17.1 lint and format checks, whitespace checks and patch-application checks.

Baseline failures were observed before each repair. F001 includes an asymmetric interaction that a simple divide-by-two fix would mishandle. F002 and F004 include unequal system sizes. F003 checks both a nonzero active interaction and finite zero forces outside support. F006 checks single and batched round trips with explicit masses.

## Apply and review a patch locally

Use a separate clean checkout for each patch. These are independent PRs against the same base, rather than a preassembled patch series:

```sh
git clone https://github.com/TorchSim/torch-sim.git torchsim-f001-review
cd torchsim-f001-review
git checkout -b brian/review-f001 d6d8e9fc30129e01a05b551c2d71912821c9b68b
# Replace this path with the location of the downloaded validation repository.
git am /path/to/torchsim-validation/patches/F001.patch
```

Install the project's development environment, then use the focused test command in that finding's PR draft. The new regression tests are included in the patch; they are absent from the original upstream checkout.

## Publication and backlinks

The evidence repository is private and the five fixes are local. [Submission status](../docs/submission.md) records the current overlap check and publication sequence. Once the evidence is public and each upstream PR exists, add its actual URL to this index and the matching finding report. The [Slack message draft](../docs/slack-message.md) describes the present evidence without implying that all findings have the same type of support.
