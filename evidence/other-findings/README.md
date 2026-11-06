# Supplemental evidence

`results.json` is produced by real CPU TorchSim 0.6.2 (audited commit), PyTorch 2.15.0 and ASE 3.29.0 calculations. `jaxmd.json` is produced in a separate native JAX MD 0.2.29 / JAX 0.12.0.1 environment. No TorchSim code or force formula is imported into that reference. Each JSON includes numeric inputs/results, scope, version information and native source checksums. No local source paths, virtual environments or caches are included.

The TorchSim collector verifies the installed distribution's VCS commit metadata, or the actual source checkout's commit and absence of tracked edits, before collecting results. It uses the shared provenance helper and records the verified result; an enclosing evidence repository is not treated as the source of a noneditable VCS install. Wrong pins, tracked source edits, or unverifiable installed commits are rejected. The JAX MD collector checks the actual installed distribution version is 0.2.29. These are CPU measurements and do not establish results on every platform.

From the repository root, with `python` resolved to the indicated isolated environment:

```sh
# Native JAX MD environment (requirements/jaxmd.txt)
python -m reproductions.native_jaxmd --output evidence/other-findings/jaxmd.json
# TorchSim + ASE environment
python -m reproductions.run_other_findings --jax-reference evidence/other-findings/jaxmd.json
# Standard-library snapshot validation (any Python)
python -m reproductions.validate evidence/other-findings/results.json --jax-reference evidence/other-findings/jaxmd.json
# Snapshot/mutation tests
python -m pytest -q tests/test_other_findings.py
# Opt-in live TorchSim/ASE recalculation test
TORCHSIM_VALIDATION_LIVE=1 python -m pytest -q tests/test_other_findings.py
```

The live runner chooses a temporary Warp cache before importing TorchSim unless `WARP_CACHE_PATH` is already set. External interpreters can run these modules directly; environment locations are not embedded in tracked code. The native ASE controls are actual `LennardJones` atom/total energy methods, `IdealGas` + independent `VelocityVerlet` runs, `Stationary(preserve_temperature=False)` and trajectory IO. F007 and F008 are same-API/state consistency checks. F005 is qualified because JAX MD shares TorchSim's projection and both projections zero total momentum.

The default exit zero means the recorded observations and controls have been validated. It does **not** mean upstream is fixed. Add `--strict` to `run_other_findings` to exit 1 after writing evidence while the observations remain. A changed observation fails validation and requires reassessing the evidence; the script never changes TorchSim.
