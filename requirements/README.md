# Isolated native environments

Snapshot validation needs only Python 3.10+ and its standard library:

```sh
python3 scripts/run_force_comparison.py
```

The native calculations ran with CPython 3.12.9 on macOS ARM64, CPU,
float64. Install Python 3.12, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/)
first. Setup was authored against uv 0.7.19. The runner creates separate ignored
`.venvs/{engine}` environments without installing into the invoking Python.

```sh
python3 scripts/run_force_comparison.py --setup --python-base python3.12
python3 scripts/run_force_comparison.py --live
```

`--setup` prepares environments; a successful setup alone does not mean a
scientific check has passed. `--setup --live` performs both steps. Use
`--setup --engines ase` for a small installation smoke check. Native collectors
can also run individually:

```sh
.venvs/ase/bin/python references/native_ase.py ase-smoke.json
```

The engine requirements files pin every dependency in the original executed
environments. ASE's file contains only its dependency closure. Setup uses
`--no-deps` to prevent the installer from silently resolving new dependency versions.
The two Git-source packages are installed from clean exact-commit checkouts in
`.venvs/sources`, with the backends pinned in `build.txt`. At runtime, the runner
verifies the actual Git commit and that tracked files are clean. For existing
noneditable VCS installs, it accepts the installed `direct_url.json` commit
metadata instead. Package version strings alone cannot establish the two source
commits.

The complete five-engine live run was verified with existing isolated native
environments. Fresh setup was smoke-tested for ASE; reinstalling all heavyweight
engines from scratch has not been verified. The exact native locks target the
recorded macOS ARM64 / Python 3.12 environment. Linux, Intel macOS, and Windows
are unverified. Platform-specific wheels, MPI libraries, and PyTorch dependencies
can require a separate lock. These platform limits do not affect snapshot
validation. LAMMPS's MPICH runtime can require permission to open a local network
interface even for this one-process, nonperiodic calculation.

To reuse installed engines without downloading anything, provide their interpreters:

```sh
python3 scripts/run_force_comparison.py --live \
  --torchsim-python "$TORCHSIM_PYTHON" \
  --ase-python "$ASE_PYTHON" \
  --lammps-python "$LAMMPS_PYTHON" \
  --jaxmd-python "$JAXMD_PYTHON" \
  --torchmd-python "$TORCHMD_PYTHON"
```

The same environment variables work without flags. The runner checks flags
first, then variables, then `.venvs/{engine}/bin/python`. `--env-dir` selects a
different isolated environment root. `--evidence-dir` selects snapshot input or
live output. Live mode always runs all five engines and writes evidence only
after the complete set passes comparison. Failed runs leave the snapshots in
place. The stdout report is JSON; progress goes to stderr.

The full consumer test suite has one test dependency, pinned separately:

```sh
uv run --with pytest==9.2.0 python -m pytest tests -q
```

Force-consumer tests also run with only the standard library:

```sh
python3 -m unittest discover -s tests -p test_force_comparison.py
```
