# Supplemental evidence

`results.json` records native CPU TorchSim 0.6.2 (audited commit), PyTorch 2.15.0 and ASE 3.29.0 calculations. `jaxmd.json` is produced in a separate native JAX MD 0.2.29 / JAX 0.12.0.1 environment. That reference imports no TorchSim code or force formula. Each JSON includes numeric inputs/results, scope, version information and native source checksums. The files exclude local source paths, virtual environments and caches.

Before collecting results, the TorchSim collector verifies the installed distribution's VCS commit metadata, or the actual source checkout's commit and absence of tracked edits. It uses the shared provenance helper and records the verified result. An enclosing evidence repository is not treated as the source of a noneditable VCS install. The collector rejects wrong pins, tracked source edits and unverifiable installed commits. The JAX MD collector checks that the actual installed distribution version is 0.2.29. These CPU measurements do not establish results on every platform.

Run these from the repository root, with `python` resolved to the indicated isolated environment:

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

The live runner chooses a temporary Warp cache before importing TorchSim unless `WARP_CACHE_PATH` is already set. External interpreters can run these modules directly; tracked code does not embed environment locations. The native ASE controls use the actual `LennardJones` atom/total energy methods, `IdealGas` + independent `VelocityVerlet` runs, `Stationary(preserve_temperature=False)` and trajectory IO. F007 and F008 check consistency within the same API/state. F005 is qualified because JAX MD shares TorchSim's projection and both projections zero total momentum.

The default exit zero means validation confirmed the recorded observations and controls. It does **not** mean upstream is fixed. Add `--strict` to `run_other_findings` to exit 1 after writing evidence while the observations remain. A changed observation fails validation and requires the evidence to be reassessed. The script never changes TorchSim.
