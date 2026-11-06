# Verification of this evidence repository

Checks recorded on November 6, 2026. The audited TorchSim source is commit `d6d8e9fc30129e01a05b551c2d71912821c9b68b`; upstream `main` matched that commit when checked.

## Executed checks

- The fresh portable force harness ran all five native libraries successfully with the documented interpreter overrides. Four external force vectors agreed within approximately `1.55e-15`; the affected TorchSim path returned twice the reference forces, and both existing TorchSim controls agreed.
- Supplemental native TorchSim/ASE calculations reran successfully in module and standalone modes. The independent native JAX MD collector also reran successfully at version 0.2.29.
- The full test suite with `TORCHSIM_VALIDATION_LIVE=1` recorded **56 passed**, plus **22 passing subtests**. It includes native supplemental reexecution, evidence-consumer mutation tests and interpreter/provenance fixtures.
- A clean exported copy of tracked files passed the lightweight force and supplemental snapshot validators. The full suite, including native supplemental calculations, also passed from that copy before the final validator hardening. After that hardening, the final full suite passed from the working repository.
- Fresh isolated ASE setup and native execution passed. A complete fresh reinstall of all five heavyweight stacks has not been executed. See [requirements/README.md](../requirements/README.md) for the platform and setup limits.
- The original TorchSim checkout's tracked implementation remained unchanged.
- Independent review identified source-provenance, relative-path and invalid-reference acceptance gaps, which were resolved. A scoped rereview found that the fixes addressed those findings.

The live suite emitted 15 dependency deprecation warnings from ASE's array-shape operations under the pinned NumPy 2.5 release. The outcomes were unchanged. The warnings are retained as a dependency compatibility observation, not a TorchSim finding.

## Recheck

```sh
python3 scripts/run_force_comparison.py
python3 -m reproductions.validate evidence/other-findings/results.json \
  --jax-reference evidence/other-findings/jaxmd.json
uv run --with pytest==9.2.0 python -m pytest tests -q
```

To run live supplemental tests, use the installed TorchSim/ASE environment for the last command and set `TORCHSIM_VALIDATION_LIVE=1`. For live five-engine computations, see the top-level README and `scripts/run_force_comparison.py --help`.

Successful evidence validation confirms the recorded discrepancies and controls. It does not establish that upstream defects are fixed or certify untested workflows or platforms.
