"""Metadata/output utilities only; no force or energy implementation."""
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform
import subprocess
import sys


CONFIGURATION = {
    "potential": "lennard_jones", "epsilon": 1.0, "sigma": 1.0,
    "cutoff": 2.5, "pbc": False, "device": "cpu", "dtype": "float64",
    "geometry": "atom 0 at [0,0,0]; atom 1 at [r,0,0]",
}


def source_provenance(module_file, package, expected_commit):
    """Read actual installed-source provenance; never infer commit from version."""
    module_file = Path(module_file).resolve()
    direct = distribution(package).read_text("direct_url.json")
    record = json.loads(direct) if direct else {}
    vcs = record.get("vcs_info")
    if vcs is not None:
        if vcs.get("commit_id") != expected_commit:
            raise ValueError(f"{package}: installed distribution VCS commit differs from pin")
        return {"commit": expected_commit, "verification": "installed distribution direct_url.json VCS commit"}
    for root in module_file.parents:
        if (root / ".git").exists():
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            # Ignore untracked runtime caches, but reject tracked source edits.
            changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
            if changed:
                raise ValueError(f"{package}: installed source has tracked changes")
            if commit != expected_commit:
                raise ValueError(f"{package}: installed Git commit differs from pin")
            return {"commit": commit, "verification": "installed Git checkout; tracked tree clean"}
    raise ValueError(f"{package}: exact installed source commit cannot be verified")


def save(data):
    """Standardize comparison units while retaining native configuration metadata."""
    data["configuration_original"] = data.get("configuration", data.get("parameters", data.get("settings", {})))
    data["configuration"] = CONFIGURATION.copy()
    data["units_original"] = data.get("units", "reduced: epsilon=sigma=1")
    data["units"] = {"distance": "sigma", "forces": "F*sigma/epsilon", "energy": "epsilon"}
    data["python"] = platform.python_version()
    data["platform"] = platform.platform()
    data["snapshot_origin"] = "native-library execution of references/ collectors"
    target = Path(sys.argv[1])
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def source_file_digest(path, relative_label):
    return {"source_file": relative_label, "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
