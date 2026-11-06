# Submission status

Prepared for Brian's review on November 6, 2026. Use Brian's formal voice in the PR descriptions and Slack message: direct wording, concrete examples and explicit limits on what was verified.

## Current state

- Five implementation changes and regression tests are prepared on separate local branches. Their patches, test results and draft descriptions are indexed in [the PR package](../pull-requests/README.md).
- All eight finding reports include recommended changes or, for F005, a clarification followed by a conditional change.
- The [high-level one-page PDF](../output/pdf/torchsim-validation-overview.pdf) has been rendered and visually checked. It contains no code or setup commands.
- The [Slack message](slack-message.md) is a draft. No message has been sent.
- No upstream PR, public branch or public disclosure has been created by this submission work. The evidence repository remains private while Brian reviews it.

## Overlap check

The audit refreshed the seven open upstream PRs through GitHub's API on November 6, 2026. The related patches below were inspected in addition to their titles:

| Open PR | Relevant work | Relationship to these findings |
|---|---|---|
| [#630](https://github.com/TorchSim/torch-sim/pull/630) | Zero-volume stress handling in pair and electrostatic models | Touches the pair-model file, but its patch does not fix F001 force accumulation, F002 energy allocation or F003 fractional powers |
| [#594](https://github.com/TorchSim/torch-sim/pull/594) | Per-system periodic boundaries | Touches the ASE exporter, but its constructor still omits explicit masses; F006 is separate and may need a small rebase if this merges first |
| [#598](https://github.com/TorchSim/torch-sim/pull/598) | Optional mass weighting for velocity autocorrelation | Does not address mass preservation in ASE conversion |

The other open PRs were #631 (autobatching memory errors), #581 (enhanced sampling), #473 (nudged elastic band) and #263 (thermodynamic integration). This refresh found no open PR implementing the five proposed repairs. It is a time-bounded check of the public open PRs, rather than a claim about all private or unpublished work. Refresh it before opening the submissions.

## Next publication step

After Brian reviews the PDF and five proposed fixes, publish the evidence repository and submit one upstream PR per finding: F001, F002, F003, F004 and F006. Each PR description already links to its evidence report. Add the actual upstream URLs back to the finding reports and the PR index, then use the Slack draft with those accessible links.

F005 stays qualified until maintainers clarify the initialization convention. F007 and F008 remain separately documented consistency findings; they are outside this five-PR implementation package.
