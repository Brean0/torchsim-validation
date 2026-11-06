"""Validate the measured discrepancies and native controls using only stdlib."""
import argparse
import json
import math
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _flatten(value):
    if isinstance(value, list):
        return [x for item in value for x in _flatten(item)]
    return [value]


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float):
        _require(math.isfinite(value), "JSON numbers must be finite; encode nonfinite observations as null")


def _close(a, b, atol=1e-11):
    a, b = _flatten(a), _flatten(b)
    return len(a) == len(b) and all(
        isinstance(x, (float, int)) and isinstance(y, (float, int))
        and math.isclose(x, y, rel_tol=1e-10, abs_tol=atol)
        for x, y in zip(a, b)
    )


def validate_evidence(document, jax_reference=None):
    """Return verified IDs or raise. This validates observations, not fixes."""
    _finite_tree(document)
    _require(document.get("schema_version") == 1, "Unsupported supplemental schema")
    _require(document.get("torchsim_commit") == "d6d8e9fc30129e01a05b551c2d71912821c9b68b", "Evidence is not from the audited TorchSim commit")
    provenance = document.get("source_provenance")
    _require(isinstance(provenance, dict)
             and provenance.get("commit") == document["torchsim_commit"]
             and provenance.get("verification") in {
                 "installed distribution direct_url.json VCS commit",
                 "installed Git checkout; tracked tree clean",
             }, "Audited installed source provenance is missing or inconsistent")
    f = document["findings"]
    _require(set(f) == {f"F{i:03d}" for i in range(2, 9)}, "Expected seven supplemental findings")

    e = f["F002"]
    _require(_close(sum(e["torchsim_per_atom"]), 2 * e["torchsim_total"])
             and not _close(sum(e["torchsim_per_atom"]), e["torchsim_total"]), "F002: expected measured factor-two energy partition discrepancy")
    _require(_close(sum(e["ase_per_atom"]), e["ase_total"]), "F002: native ASE energy sum invariant failed")
    original = e["original_soft_sphere_regression"]
    _require(_close(original["total"], 0.125) and _close(sum(original["per_atom"]), 0.25), "F002: original soft-sphere regression not reproduced")

    e = f["F003"]
    expected_soft_inputs = {
        "species": [0, 0, 1], "epsilon": 1.0, "alpha": 2.5,
        "sigma_matrix": [[1.0, 2.0], [2.0, 2.0]],
        "positions": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
    }
    for name, expected in expected_soft_inputs.items():
        _require(e.get(name) == expected, f"F003: TorchSim missing or non-equivalent {name}")
    flat = _flatten(e["forces"])
    _require(len(flat) == 9 and flat.count(None) == e["nonfinite_force_components"] > 0,
             "F003: expected nonfinite forces, explicitly counted")
    _require(_close(e["energy"], [0.0]), "F003: outside-support energy must remain zero")
    _require(_close(e["integer_alpha_control_forces"], [[0.0] * 3] * 3), "F003: integer-alpha control failed")

    e = f["F004"]
    expected = [[dt] * 3 for dt in e["dt"] for _ in range(2)]
    _require(_close(e["expected_positions"], expected) and _close(e["ase_positions"], expected), "F004: native per-system integration control failed")
    error = max(abs(x-y) for x, y in zip(_flatten(e["torchsim_positions"]), _flatten(expected)))
    _require(error > 0.1 and _close(error, e["max_position_error"]), "F004: expected observed vector timestep discrepancy")
    _require(_close(e["scalar_timestep_max_errors"], [0.0] * 3), "F004: scalar controls failed")

    e = f["F005"]
    raw, masses = e["raw_momenta"], e["masses"]
    total = [sum(p[k] for p in raw) for k in range(3)]
    mean_projection = [[p[k]-total[k]/len(raw) for k in range(3)] for p in raw]
    velocity_projection = [[p[k]-m*total[k]/sum(masses) for k in range(3)] for p, m in zip(raw, masses)]
    _require(_close(e["torchsim_momenta"], mean_projection), "F005: TorchSim mean-momentum projection not reproduced")
    _require(_close(e["ase_momenta"], velocity_projection) and _close(e["ase_projection_error"], 0.0), "F005: ASE common-velocity projection control failed")
    _require(not _close(e["torchsim_momenta"], e["ase_momenta"]), "F005: unequal-mass projections unexpectedly agree")
    _require(_close(e["torchsim_total_momentum"], [0.0] * 3) and _close(e["ase_total_momentum"], [0.0] * 3), "F005: both projections must remove total momentum")
    _require(_close(e["torchsim_ensemble_variance"], e["mean_projection_variance"], atol=0.05), "F005: observed TorchSim ensemble contradicts projection variance")

    e = f["F006"]
    _require(_close(e["input_masses"], [2.014, 2.014]) and _close(e["state_masses"], e["input_masses"]), "F006: custom mass import control failed")
    _require(_close(e["native_ase_roundtrip_masses"], e["input_masses"]), "F006: native ASE mass-preserving IO control failed")
    _require(not _close(e["exported_masses"], e["input_masses"])
             and _close(e["exported_masses"], e["ase_default_H_masses"]), "F006: expected loss of explicit masses")

    e = f["F007"]
    _require(e["input_dtype"] == e["list_dtype"] == "torch.float32"
             and e["requested_dtype"] == e["single_dtype"] == "torch.float64", "F007: list/single dtype discrepancy not reproduced")
    _require(e["requested_device"] == e["list_device"] == e["single_device"] == "cpu", "F007: this record only supports CPU dtype evidence")

    e = f["F008"]
    _require(e["proposed_energy_delta"] > 0 and e["acceptance_probability"] == 0.0, "F008: proposal must be deterministically rejected at tested kT")
    _require(_close(e["original_positions"], e["returned_positions"])
             and _close(e["original_energy"], e["returned_energy"]), "F008: rejected state rollback control failed")
    _require(not _close(e["returned_signal"], e["reevaluated_signal"])
             and _close(e["original_signal"], e["reevaluated_signal"]), "F008: extras inconsistency not reproduced")

    if jax_reference is not None:
        _finite_tree(jax_reference)
        _require(jax_reference["engine"] == "JAX MD" and jax_reference["version"] == "0.2.29", "Unexpected JAX MD reference")
        j = jax_reference["fractional_soft_sphere"]
        for name, expected in expected_soft_inputs.items():
            _require(j.get(name) == expected, f"F003: JAX MD missing or non-equivalent {name}")
        _require(_close(j["positions"], f["F003"]["positions"])
                 and _close(j["energy"], 0.0)
                 and _close(j["forces"], [[0.0] * 3] * 3), "F003: native JAX MD derivative control failed")
        j = jax_reference["unequal_mass_initialization"]
        raw = j["raw_momenta"]
        total = [sum(p[k] for p in raw) for k in range(3)]
        projected = [[p[k]-total[k]/len(raw) for k in range(3)] for p in raw]
        _require(_close(j["initialized_momenta"], projected)
                 and j["mean_projection_max_error"] < 1e-11
                 and j["ase_style_projection_max_error"] > 0.01
                 and _close(j["total_momentum"], [0.0] * 3), "F005: native JAX MD qualification failed")
    return sorted(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--jax-reference", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    reference = json.loads(args.jax_reference.read_text()) if args.jax_reference else None
    print(json.dumps({"validated_observations": validate_evidence(evidence, reference),
                      "meaning": "Recorded discrepancies and native controls reproduced; upstream is not fixed."}))


if __name__ == "__main__":
    main()
