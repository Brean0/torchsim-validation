# Isolated native environments

Snapshot validation needs Python 3.10+ and its standard library only:

```sh
python3 scripts/run_force_comparison.py
```

The executed native environments used CPython 3.12.9 on macOS ARM64, CPU,
float64. Install Python 3.12, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/)
first. Setup was authored against uv 0.7.19. The runner creates separate ignored
`.venvs/{engine}` environments; it never installs into the invoking Python.

```sh
python3 scripts/run_force_comparison.py --setup --python-base python3.12
python3 scripts/run_force_comparison.py --live
```

`--setup` only prepares environments; it does not claim a scientific check has
passed. `--setup --live` performs both steps. A small installation smoke check is
available with `--setup --engines ase`. Native collectors can be run individually:

```sh
.venvs/ase/bin/python references/native_ase.py ase-smoke.json
```

Every dependency in the original executed environments is pinned in an engine
requirements file. ASE's file contains only its dependency closure. Setup uses
`--no-deps` so the installer cannot silently resolve new dependency versions.
The two Git-source packages are installed from clean exact-commit checkouts in
`.venvs/sources`, with the backends pinned in `build.txt`. Runtime verifies the
actual Git commit and that tracked files are clean. For existing noneditable VCS
installs it accepts the installed `direct_url.json` commit metadata instead.
Package version strings alone cannot establish the two source commits.

The complete five-engine live run is verified using existing isolated native
environments. Fresh setup is smoke-tested for ASE. Reinstalling all heavyweight
engines from scratch has not been verified. The exact native locks target the
recorded macOS ARM64 / Python 3.12 environment. Linux, Intel macOS, and Windows
are unverified; platform-specific wheels, MPI libraries, and PyTorch dependencies
can require a separate lock. Snapshot validation is independent of those
platform limitations. LAMMPS's MPICH runtime can require permission to open a
local network interface even for this one-process, nonperiodic calculation.

To reuse installed engines without downloading anything, supply interpreters:

```sh
python3 scripts/run_force_comparison.py --live \
  --torchsim-python "$TORCHSIM_PYTHON" \
  --ase-python "$ASE_PYTHON" \
  --lammps-python "$LAMMPS_PYTHON" \
  --jaxmd-python "$JAXMD_PYTHON" \
  --torchmd-python "$TORCHMD_PYTHON"
```

The same environment variables also work without flags. Flags take precedence
over variables, followed by `.venvs/{engine}/bin/python`. `--env-dir` selects a
different isolated environment root. `--evidence-dir` selects snapshot input or
live output. Live mode always runs all five engines and writes evidence only
after the complete set passes comparison; failed runs do not replace snapshots.
The stdout report is JSON; progress is sent to stderr.

For the full consumer test suite, the only test dependency is pinned separately:

```sh
uv run --with pytest==9.2.0 python -m pytest tests -q
```

Force-consumer tests also run with only the standard library:

```sh
python3 -m unittest discover -s tests -p test_force_comparison.py
```
