#!/usr/bin/env python3
"""Validate saved native evidence (default), or set up/rerun isolated engines.

Exit zero verifies the recorded F001 observation and its controls. It does not
mean that the audited TorchSim defect has been fixed. Snapshot mode uses only
the Python standard library and never downloads or executes a native engine.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("torchsim", "ase", "lammps", "jaxmd", "torchmd")
DISTANCES = (1.1, 1.3, 1.5)
COMMIT = "d6d8e9fc30129e01a05b551c2d71912821c9b68b"
TORCHMD_COMMIT = "09484183e34af78bb69b27a0149aeccf1318dbc7"
VERSIONS = {"torchsim": "0.6.2", "ase": "3.29.0", "lammps": 20250722,
            "jaxmd": "0.2.29", "torchmd": "0.1.dev1+g09484183e"}
UNITS = {"distance": "sigma", "forces": "F*sigma/epsilon", "energy": "epsilon"}
CONFIGURATION = {
    "potential": "lennard_jones", "epsilon": 1.0, "sigma": 1.0,
    "cutoff": 2.5, "pbc": False, "device": "cpu", "dtype": "float64",
    "geometry": "atom 0 at [0,0,0]; atom 1 at [r,0,0]",
}
PATHS = {
    "PairForcesModel_full_default": 2.0,
    "PairForcesModel_half": 1.0,
    "PairPotentialModel": 1.0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, label):
    """Reject JSON non-finite numbers everywhere, including optional energies."""
    if isinstance(value, float):
        require(math.isfinite(value), f"{label}: nonfinite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{label}[{index}]")


def vector(value, label):
    """Validate forces for the isolated central-force pair on the x axis."""
    require(isinstance(value, list) and len(value) == 2, f"{label}: expected 2x3 force array")
    for row in value:
        require(isinstance(row, list) and len(row) == 3, f"{label}: expected 2x3 force array")
        for number in row:
            require(type(number) in (int, float) and math.isfinite(number), f"{label}: invalid force")
        require(all(math.isclose(number, 0.0, rel_tol=0.0, abs_tol=5e-12) for number in row[1:]),
                f"{label}: transverse force is incompatible with the x-axis pair")
    require(math.isclose(value[0][0], -value[1][0], rel_tol=1e-12, abs_tol=5e-12),
            f"{label}: atom forces must be equal and opposite")
    return value


def close(actual, expected, label, factor=1.0):
    actual = vector(actual, label)
    expected = vector(expected, label + " expected")
    maximum = 0.0
    for row_a, row_b in zip(actual, expected):
        for a, b in zip(row_a, row_b):
            target = factor * b
            require(math.isclose(a, target, rel_tol=1e-12, abs_tol=5e-12),
                    f"{label}: {a!r} != {target!r}")
            maximum = max(maximum, abs(a - target))
    return maximum


def cases_by_distance(cases, label, expected=DISTANCES):
    require(isinstance(cases, list), f"{label}: missing case list")
    selected = {}
    for case in cases:
        require(isinstance(case, dict), f"{label}: case must be an object")
        distance = case.get("distance")
        require(type(distance) in (int, float) and math.isfinite(distance), f"{label}: invalid distance")
        require(distance not in selected, f"{label}: duplicate distance {distance}")
        selected[distance] = case
    require(set(selected) == set(expected), f"{label}: expected distances {list(expected)}")
    return selected


def validate(raw, fresh=False):
    """Compare real recorded vectors, without reimplementing any pair force."""
    for name in ENGINES:
        data = raw[name]
        require(isinstance(data, dict), f"{name}: expected object")
        finite_tree(data, name)
        require(bool(data.get("engine")) and data.get("version") is not None, f"{name}: engine/version missing")
        require(data["version"] == VERSIONS[name], f"{name}: installed version differs from pin")
        require(data.get("configuration") == CONFIGURATION, f"{name}: non-equivalent configuration")
        require(data.get("units") == UNITS, f"{name}: non-equivalent units")
    require(raw["torchsim"].get("commit") == COMMIT, "TorchSim source commit does not match audit")
    require(raw["torchmd"].get("source_commit") == TORCHMD_COMMIT, "TorchMD source commit does not match audit")
    references = {name: cases_by_distance(raw[name]["cases"], name) for name in ENGINES[1:]}
    baseline = references["ase"]
    for distance in DISTANCES:
        force = vector(baseline[distance]["forces"], f"ASE r={distance}")
        require(abs(force[0][0]) > 1e-6, f"r={distance}: force is too small to test doubling")
        require(force[0][0] < 0 if distance == 1.1 else force[0][0] > 0,
                f"r={distance}: unexpected force direction for the recorded geometry")
    maximum = 0.0
    rows = {}
    for name, cases in references.items():
        rows[name] = []
        for distance in DISTANCES:
            force = cases[distance]["forces"]
            maximum = max(maximum, close(force, baseline[distance]["forces"], f"{name} r={distance}"))
            rows[name].append(force[0][0])
    ts_lj = cases_by_distance(
        [case for case in raw["torchsim"]["cases"] if case.get("potential") == "lennard_jones"], "TorchSim LJ")
    for path, factor in PATHS.items():
        rows[path] = []
        for distance in DISTANCES:
            for backend in ("default", "torch_nl_n2"):
                close(ts_lj[distance]["forces"][f"{path}/{backend}"], baseline[distance]["forces"],
                      f"TorchSim {path}/{backend} r={distance}", factor)
            rows[path].append(ts_lj[distance]["forces"][path + "/default"][0][0])
    physical = cases_by_distance(raw["torchmd"]["physical_epsilon_1ev_cases"], "TorchMD physical units")
    for distance in DISTANCES:
        close(physical[distance]["forces_ev_angstrom"], baseline[distance]["forces"], "TorchMD 1 eV conversion")
    soft_ref = raw["jaxmd"]["original_soft_sphere_control"]
    require(soft_ref.get("distance") == 0.5 and soft_ref.get("alpha") == 2.0,
            "original soft-sphere configuration differs")
    # This literal is the original alpha=2, r=0.5 control observation, not a
    # replacement force computation. It prevents vacuous or coherently wrong
    # soft-sphere evidence from being reported as a verified control.
    close(soft_ref["forces"], [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
          "original soft-sphere native control")
    soft = cases_by_distance(
        [case for case in raw["torchsim"]["cases"] if case.get("potential") == "soft_sphere_alpha_2"],
        "TorchSim soft sphere", (0.5,))[0.5]
    for path, factor in PATHS.items():
        for backend in ("default", "torch_nl_n2"):
            close(soft["forces"][f"{path}/{backend}"], soft_ref["forces"], f"soft sphere {path}/{backend}", factor)
    return {
        "finding": "F001", "status": "confirmed_observation", "fresh_engine_runs": fresh,
        "scope": "Default full-list PairForcesModel doubling for this isolated pair; controls verified",
        "torchsim_commit": COMMIT, "distances": list(DISTANCES), "units": UNITS,
        "atom_0_x_force": rows, "external_reference_count": 4,
        "maximum_reference_difference": maximum,
        "reference_agreement_tolerance": {"rtol": 1e-12, "atol": 5e-12},
        "default_pairforces_factor": 2.0,
        "both_torchsim_controls_match_references": True,
        "both_torchsim_neighbor_backends_agree": True,
        "original_soft_sphere_controls_verified": True,
        "raw_result_files": {name: f"{name}.json" for name in ENGINES},
        "interpretation": "Exit zero confirms the stated observation; upstream is not fixed.",
    }


def run(command, **kwargs):
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def setup(engines, args):
    uv = shutil.which("uv")
    require(uv is not None, "--setup requires uv; see requirements/README.md")
    # Refuse to silently substitute a different minor Python version.
    version = subprocess.check_output([args.python_base, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"], text=True).strip()
    require(version == "3.12", "live setup requires Python 3.12 (--python-base selects it)")
    for engine in engines:
        destination = args.env_dir / engine
        if not (destination / "bin/python").exists():
            run([uv, "venv", "--python", args.python_base, destination])
        python = destination / "bin/python"
        run([uv, "pip", "install", "--python", python, "--no-deps", "-r", ROOT / "requirements" / f"{engine}.txt"])
        if engine in ("torchsim", "torchmd"):
            source = args.env_dir / "sources" / engine
            source.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/torchsim/torch-sim.git" if engine == "torchsim" else "https://github.com/torchmd/torchmd.git"
            commit = COMMIT if engine == "torchsim" else TORCHMD_COMMIT
            if not (source / ".git").exists():
                run(["git", "clone", "--no-checkout", url, source])
            run(["git", "-C", source, "checkout", "--detach", commit])
            # Pin build tools too, and retain Git for runtime source provenance.
            run([uv, "pip", "install", "--python", python, "--no-deps", "-r", ROOT / "requirements/build.txt"])
            run([uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation", "-e", source])
        print(f"Prepared {engine} environment", file=sys.stderr)


def live(args):
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    interpreters = {}
    for engine in ENGINES:
        selected = getattr(args, f"{engine}_python") or os.environ.get(engine.upper() + "_PYTHON") or args.env_dir / engine / "bin/python"
        selected = os.path.expanduser(str(selected))
        if not os.path.isabs(selected) and os.sep not in selected:
            selected = shutil.which(selected)
            require(selected is not None, f"{engine}: interpreter command was not found")
        # Make paths absolute against the invocation cwd before the child gets
        # a temporary cwd. Preserve a venv's final python symlink: resolve()
        # would turn it into its base interpreter and discard the environment.
        interpreters[engine] = os.path.abspath(selected)
    with tempfile.TemporaryDirectory(prefix="torchsim-validation-") as directory:
        environment = os.environ.copy()
        environment["WARP_CACHE_PATH"] = str(Path(directory) / "warp-cache")
        environment["JAX_PLATFORMS"] = "cpu"
        environment["JAX_ENABLE_X64"] = "true"
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        for engine in ENGINES:
            python = interpreters[engine]
            target = Path(directory) / f"{engine}.json"
            print(f"Running native {engine} calculation", file=sys.stderr, flush=True)
            process = subprocess.run([str(python), str(ROOT / "references" / f"native_{engine}.py"), str(target)],
                                     cwd=directory, env=environment, text=True, capture_output=True)
            require(process.returncode == 0, f"{engine} native run exited {process.returncode}: {process.stderr[-6000:]}")
            require(target.exists(), f"{engine} produced no evidence")
        raw = {engine: json.loads((Path(directory) / f"{engine}.json").read_text()) for engine in ENGINES}
        report = validate(raw, fresh=True)
        # Publish only a complete, validated five-engine set; logs stay temporary.
        for engine, data in raw.items():
            (args.evidence_dir / f"{engine}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        (args.evidence_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="rerun all five native engines and save validated evidence")
    parser.add_argument("--setup", action="store_true", help="create isolated pinned environments with uv; use --live too to run")
    parser.add_argument("--engines", default=",".join(ENGINES), help="comma-separated setup subset; --live always runs all five")
    parser.add_argument("--python-base", default="python3.12", help="Python 3.12 executable used for setup")
    parser.add_argument("--env-dir", type=Path, default=ROOT / ".venvs")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence/force-doubling")
    for engine in ENGINES:
        parser.add_argument(f"--{engine}-python", help=f"existing {engine} interpreter; also {engine.upper()}_PYTHON")
    args = parser.parse_args(argv)
    try:
        engines = args.engines.split(",")
        require(all(name in ENGINES for name in engines), "--engines contains an unknown engine")
        if args.setup:
            setup(engines, args)
            if not args.live:
                return 0
        if args.live:
            report = live(args)
        else:
            raw = {engine: json.loads((args.evidence_dir / f"{engine}.json").read_text()) for engine in ENGINES}
            report = validate(raw)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
