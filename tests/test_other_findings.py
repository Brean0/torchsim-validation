"""Check measured observations; success does not imply TorchSim is fixed."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reproductions.validate import validate_evidence


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    return json.loads((ROOT / "evidence/other-findings/results.json").read_text())


def test_saved_real_calculations_validate(evidence):
    reference = json.loads((ROOT / "evidence/other-findings/jaxmd.json").read_text())
    assert validate_evidence(evidence, reference) == [f"F{i:03d}" for i in range(2, 9)]


@pytest.mark.parametrize("finding,key,value", [
    ("F002", "torchsim_total", 99.0),
    ("F003", "nonfinite_force_components", 0),
    ("F004", "max_position_error", 0.0),
    ("F005", "ase_projection_error", 1.0),
    ("F006", "exported_masses", [2.014, 2.014]),
    ("F007", "list_dtype", "torch.float64"),
    ("F008", "reevaluated_signal", [2.0, 0.0]),
])
def test_rejects_changed_or_fabricated_observation(evidence, finding, key, value):
    invalid = copy.deepcopy(evidence)
    invalid["findings"][finding][key] = value
    with pytest.raises(ValueError):
        validate_evidence(invalid)


def test_rejects_nonfinite_json_number(evidence):
    evidence["findings"]["F002"]["ase_total"] = float("nan")
    with pytest.raises(ValueError):
        validate_evidence(evidence)


def test_rejects_missing_installed_source_verification(evidence):
    evidence.pop("source_provenance")
    with pytest.raises(ValueError, match="provenance"):
        validate_evidence(evidence)


def test_rejects_inconsistent_installed_source_commit(evidence):
    evidence["source_provenance"]["commit"] = "b" * 40
    with pytest.raises(ValueError, match="provenance"):
        validate_evidence(evidence)


def test_rejects_failed_jax_control(evidence):
    reference = json.loads((ROOT / "evidence/other-findings/jaxmd.json").read_text())
    reference["fractional_soft_sphere"]["forces"][0][0] = 1.0
    with pytest.raises(ValueError):
        validate_evidence(evidence, reference)


def test_rejects_unpinned_actual_jax_version(evidence):
    reference = json.loads((ROOT / "evidence/other-findings/jaxmd.json").read_text())
    reference["version"] = "0.2.28"
    with pytest.raises(ValueError, match="JAX MD"):
        validate_evidence(evidence, reference)


@pytest.mark.parametrize("side,field,bad", [
    ("jax", "epsilon", 0.0),
    ("jax", "species", [0, 0, 0]),
    ("jax", "alpha", 2.0),
    ("jax", "sigma_matrix", [[2.0, 2.0], [2.0, 2.0]]),
    ("jax", "positions", [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    ("torchsim", "epsilon", 0.0),
    ("torchsim", "species", [0, 0, 0]),
    ("torchsim", "alpha", 2.0),
    ("torchsim", "sigma_matrix", [[2.0, 2.0], [2.0, 2.0]]),
    ("torchsim", "positions", [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [5.0, 0.0, 0.0]]),
])
def test_rejects_non_equivalent_fractional_potential(evidence, side, field, bad):
    reference = json.loads((ROOT / "evidence/other-findings/jaxmd.json").read_text())
    target = reference["fractional_soft_sphere"] if side == "jax" else evidence["findings"]["F003"]
    target[field] = bad
    with pytest.raises(ValueError, match="F003"):
        validate_evidence(evidence, reference)


@pytest.mark.parametrize("side", ["jax", "torchsim"])
@pytest.mark.parametrize("field", ["epsilon", "species", "alpha", "sigma_matrix", "positions"])
def test_rejects_missing_fractional_potential_metadata(evidence, side, field):
    reference = json.loads((ROOT / "evidence/other-findings/jaxmd.json").read_text())
    target = reference["fractional_soft_sphere"] if side == "jax" else evidence["findings"]["F003"]
    del target[field]
    with pytest.raises(ValueError, match="F003"):
        validate_evidence(evidence, reference)


def _native_distribution(directory, version="1.0", direct_url=None, package="audit_fake_native"):
    source = directory / package / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Real fixture package for provenance consumer tests.\n")
    metadata = directory / f"{package}-{version}.dist-info"
    metadata.mkdir()
    metadata.joinpath("METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {package.replace('_', '-')}\nVersion: {version}\n"
    )
    if direct_url is not None:
        metadata.joinpath("direct_url.json").write_text(json.dumps(direct_url))
    return source


def _consume_source(site, source, expected_commit):
    code = (
        "import json, sys; sys.path[:0] = sys.argv[1:3]; "
        "from reproductions.run_other_findings import verify_source; "
        "print(json.dumps(verify_source(sys.argv[3], 'audit-fake-native', sys.argv[4])))"
    )
    return subprocess.run([sys.executable, "-c", code, str(ROOT), str(site), str(source), expected_commit],
                          capture_output=True, text=True, check=False)


def test_supplemental_consumer_prefers_real_vcs_distribution_metadata(tmp_path):
    expected = "a" * 40
    site = tmp_path / "site"
    source = _native_distribution(site, direct_url={
        "url": "https://example.invalid/native.git",
        "vcs_info": {"vcs": "git", "commit_id": expected},
    })
    # A noneditable VCS installation can be under an unrelated repository.
    # Its installed metadata must win over this broken ancestor Git directory.
    (tmp_path / ".git").mkdir()
    result = _consume_source(site, source, expected)
    assert result.returncode == 0, result.stderr
    provenance = json.loads(result.stdout)
    assert provenance["commit"] == expected
    assert "direct_url.json" in provenance["verification"]


def test_supplemental_consumer_rejects_wrong_vcs_install_pin(tmp_path):
    site = tmp_path / "site"
    source = _native_distribution(site, direct_url={
        "url": "https://example.invalid/native.git",
        "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
    })
    result = _consume_source(site, source, "b" * 40)
    assert result.returncode != 0
    assert "VCS commit differs from pin" in result.stderr


def test_supplemental_consumer_rejects_dirty_tracked_checkout(tmp_path):
    source = _native_distribution(tmp_path)
    for args in (["init", "-q"], ["add", str(source.relative_to(tmp_path))],
                 ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    clean = _consume_source(tmp_path, source, commit)
    assert clean.returncode == 0, clean.stderr
    assert json.loads(clean.stdout)["verification"] == "installed Git checkout; tracked tree clean"
    source.write_text("# Modified tracked installed source.\n")
    dirty = _consume_source(tmp_path, source, commit)
    assert dirty.returncode != 0
    assert "tracked changes" in dirty.stderr


def test_supplemental_entrypoint_supports_module_and_standalone():
    commands = [[sys.executable, "-m", "reproductions.run_other_findings", "--help"],
                [sys.executable, str(ROOT / "reproductions/run_other_findings.py"), "--help"]]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "--strict" in result.stdout


@pytest.mark.parametrize("version,passes", [("0.2.29", True), ("0.2.28", False)])
def test_native_jax_consumer_requires_actual_installed_version(tmp_path, version, passes):
    _native_distribution(tmp_path, version=version, package="jax_md")
    code = (
        "import sys; sys.path[:0] = sys.argv[1:3]; "
        "from reproductions.native_jaxmd import installed_jaxmd_version; "
        "print(installed_jaxmd_version())"
    )
    result = subprocess.run([sys.executable, "-c", code, str(ROOT), str(tmp_path)], capture_output=True, text=True)
    assert (result.returncode == 0) is passes, result.stderr
    if passes:
        assert result.stdout.strip() == "0.2.29"
    else:
        assert "expected 0.2.29" in result.stderr


@pytest.mark.skipif(os.environ.get("TORCHSIM_VALIDATION_LIVE") != "1", reason="Opt in to installed native TorchSim/ASE calculations")
def test_repeat_live_native_calculations():
    from reproductions.run_other_findings import run
    assert validate_evidence(run()) == [f"F{i:03d}" for i in range(2, 9)]
