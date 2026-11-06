# Submission status

Published on November 6, 2026 after Brian reviewed and approved the findings and proposed fixes.

The [validation repository](https://github.com/Brean0/torchsim-validation) is public. Five separate upstream PRs are open: F001, F002, F003, F004 and F006. Their URLs, patches and validation results are recorded in the [finding reports](findings/) and [findings.json](../findings.json). Each PR links back to its report.

The draft PR folder has been removed; the findings are the maintained source for submission details. All eight reports retain their recommended changes or, for F005, a clarification followed by a conditional change. The [one-page overview](../output/pdf/torchsim-validation-overview.pdf) provides the high-level summary.

The [Slack message](slack-message.md) remains a draft.

## Overlap check

The audit refreshed the seven open upstream PRs through GitHub's API on November 6, 2026. The related patches below were inspected in addition to their titles:

| Open PR | Relevant work | Relationship to these findings |
|---|---|---|
| [#630](https://github.com/TorchSim/torch-sim/pull/630) | Zero-volume stress handling in pair and electrostatic models | Touches the pair-model file, but its patch does not fix F001 force accumulation, F002 energy allocation or F003 fractional powers |
| [#594](https://github.com/TorchSim/torch-sim/pull/594) | Per-system periodic boundaries | Touches the ASE exporter, but its constructor still omits explicit masses; F006 is separate and may need a small rebase if this merges first |
| [#598](https://github.com/TorchSim/torch-sim/pull/598) | Optional mass weighting for velocity autocorrelation | Does not address mass preservation in ASE conversion |

The other open PRs were #631 (autobatching memory errors), #581 (enhanced sampling), #473 (nudged elastic band) and #263 (thermodynamic integration). Before these submissions, this refresh found no open PR implementing the five proposed repairs. It is a time-bounded check of the public open PRs, rather than a claim about all private or unpublished work. The five PRs were opened after this check.

## Other findings

F005 stays qualified until maintainers clarify the initialization convention. F007 and F008 remain separately documented consistency findings; they are outside this five-PR implementation package. An open PR is a proposed fix, not a merged upstream correction.
