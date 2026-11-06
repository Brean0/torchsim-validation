"""Run seven supplemental observations with native ASE controls, CPU float64.

Run from the repository root with the TorchSim/ASE interpreter:
    python -m reproductions.run_other_findings
The default success criterion validates the *observations*. --strict exits 1
while observed discrepancies remain; it does not modify TorchSim.
"""
import argparse
import importlib.metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

if not __package__:
    # Direct script execution puts reproductions/, rather than the repository
    # root, on sys.path. Resolve sibling shared helpers from this file location.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from references.common import source_provenance
from reproductions.validate import validate_evidence


AUDITED_COMMIT = "d6d8e9fc30129e01a05b551c2d71912821c9b68b"


def verify_source(module_file, package="torch-sim-atomistic", expected_commit=AUDITED_COMMIT):
    """Use installed VCS metadata or a clean checkout; reject unverifiable pins."""
    return source_provenance(module_file, package, expected_commit)


def run():
    # Set before importing Warp via TorchSim; do not rely on a writable home cache.
    os.environ.setdefault("WARP_CACHE_PATH", str(Path(tempfile.gettempdir()) / "torchsim-validation-warp"))
    import numpy as np
    import torch
    import torch_sim as ts
    provenance = verify_source(ts.__file__)
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.calculators.idealgas import IdealGas
    from ase.io import read, write
    from ase.md.velocitydistribution import Stationary
    from ase.md.verlet import VelocityVerlet
    from torch_sim.integrators.md import MDState, initialize_momenta
    from torch_sim.models.interface import ModelInterface
    from torch_sim.models.lennard_jones import LennardJonesModel
    from torch_sim.models.soft_sphere import SoftSphereModel, SoftSphereMultiModel
    from torch_sim.neighbors import torch_nl_n2

    dtype = torch.float64

    def state(positions, numbers, dtype=dtype):
        return ts.SimState(positions=torch.tensor(positions, dtype=dtype),
                           masses=torch.ones(len(positions), dtype=dtype),
                           cell=10 * torch.eye(3, dtype=dtype).unsqueeze(0),
                           pbc=False, atomic_numbers=torch.tensor(numbers, dtype=torch.long))

    class ZeroModel(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = dtype
            self._compute_forces = True
            self._compute_stress = True

        def forward(self, state, **kwargs):
            return {"energy": torch.zeros(state.n_systems, dtype=dtype),
                    "forces": torch.zeros_like(state.positions),
                    "stress": torch.zeros_like(state.cell)}

    class SiteSignalModel(ZeroModel):
        def forward(self, state, **kwargs):
            output = super().forward(state)
            signal = state.atomic_numbers * state.positions[:, 0]
            output["energy"] = torch.zeros(state.n_systems, dtype=dtype).index_add(0, state.system_idx, -signal)
            output["site_signal"] = signal
            return output

    def values(tensor):
        return tensor.detach().cpu().tolist()

    def sanitized(tensor):
        a = tensor.detach().cpu().numpy()
        return np.where(np.isfinite(a), a, None).tolist()

    findings = {}
    positions = [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]
    lj = LennardJonesModel(sigma=1.0, epsilon=1.0, cutoff=2.5, dtype=dtype,
                          per_atom_energies=True, neighbor_list_fn=torch_nl_n2)
    output = lj(state(positions, [18, 18]))
    atoms = Atoms("Ar2", positions=positions, pbc=False)
    atoms.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False)
    findings["F002"] = {
        "status": "observed_internal_invariant_failure_with_native_control",
        "scope": "LennardJonesModel with full torch_nl_n2; compare each engine internally, since ASE shifts cutoff energy",
        "positions": positions, "cutoff": 2.5, "sigma": 1.0, "epsilon": 1.0,
        "torchsim_total": float(output["energy"].sum()),
        "torchsim_per_atom": values(output["energies"]),
        "ase_total": float(atoms.get_potential_energy()),
        "ase_per_atom": atoms.get_potential_energies().tolist(),
        "reference_invariant": "sum(per_atom_energies) == total_energy within each engine",
        "units": "eV (sigma in angstrom)",
    }
    original_soft = SoftSphereModel(per_atom_energies=True, dtype=dtype,
                                   neighbor_list_fn=torch_nl_n2)(state([[0, 0, 0], [0.5, 0, 0]], [18, 18]))
    findings["F002"]["original_soft_sphere_regression"] = {
        "distance": 0.5, "sigma": 1.0, "epsilon": 1.0, "alpha": 2.0,
        "total": float(original_soft["energy"].sum()), "per_atom": values(original_soft["energies"]),
    }

    positions3 = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]]
    s3 = state(positions3, [18, 18, 36])
    def soft(alpha):
        return SoftSphereMultiModel(atomic_numbers=s3.atomic_numbers,
                  sigma_matrix=torch.tensor([[1.0, 2.0], [2.0, 2.0]], dtype=dtype),
                  alpha_matrix=torch.full((2, 2), alpha, dtype=dtype),
                  neighbor_list_fn=torch_nl_n2, dtype=dtype)(s3)
    output3, control3 = soft(2.5), soft(2.0)
    findings["F003"] = {
        "status": "observed_nonfinite_derivative_native_jax_reference_separate",
        "scope": "fractional alpha and species pair beyond its sigma, but inside global neighbor cutoff",
        "positions": positions3, "species": [0, 0, 1], "alpha": 2.5,
        "sigma_matrix": [[1.0, 2.0], [2.0, 2.0]], "epsilon": 1.0,
        "energy": values(output3["energy"]), "forces": sanitized(output3["forces"]),
        "nonfinite_force_components": int((~torch.isfinite(output3["forces"])).sum()),
        "integer_alpha_control_forces": values(control3["forces"]),
        "reference_invariant": "all pair distances exceed species sigma, so energies and derivatives are zero",
        "nonfinite_encoding": "null, with explicit count; strict JSON contains no NaN token",
    }

    batched = ts.SimState(positions=torch.zeros((6, 3), dtype=dtype),
                         masses=torch.ones(6, dtype=dtype),
                         cell=torch.eye(3, dtype=dtype).repeat(3, 1, 1), pbc=False,
                         atomic_numbers=torch.tensor([1, 2] * 3),
                         system_idx=torch.arange(3).repeat_interleave(2))
    md = MDState.from_state(batched, momenta=torch.ones_like(batched.positions),
                           energy=torch.zeros(3, dtype=dtype), forces=torch.zeros_like(batched.positions))
    dt = torch.tensor([0.1, 0.2, 0.3], dtype=dtype)
    expected = dt[batched.system_idx, None].expand_as(batched.positions).clone()
    output4 = ts.nve_step(md.clone(), ZeroModel(), dt=dt)
    native_positions = []
    scalar_errors = []
    for timestep in dt.tolist():
        a = Atoms("HHe", positions=np.zeros((2, 3)), masses=[1.0, 1.0], pbc=False)
        a.calc = IdealGas()
        a.set_velocities(np.ones((2, 3)))
        # timestep and velocities are in ASE's internal time units. No fs
        # conversion: both native equations evaluate displacement v_internal*dt.
        VelocityVerlet(a, timestep=timestep, logfile=None).run(1)
        native_positions.extend(a.positions.tolist())
        scalar_out = ts.nve_step(md.clone(), ZeroModel(), dt=timestep)
        scalar_errors.append(float((scalar_out.positions - timestep).abs().max()))
    findings["F004"] = {
        "status": "observed_per_system_broadcast_error_with_native_control",
        "scope": "three systems, two atoms each, zero forces, dt vector length three",
        "dt": dt.tolist(), "initial_velocity": [1.0, 1.0, 1.0],
        "torchsim_positions": values(output4.positions), "expected_positions": values(expected),
        "ase_positions": native_positions,
        "max_position_error": float((output4.positions - expected).abs().max()),
        "scalar_timestep_max_errors": scalar_errors,
        "units": "consistent internal-unit equation dr = v*dt; ASE uses its internal time (not femtoseconds)",
        "reference_invariant": "independent native ASE IdealGas + VelocityVerlet per timestep; each coordinate in one system has same displacement",
    }

    masses = torch.tensor([1.0, 4.0], dtype=dtype)
    zero_pos = torch.zeros((2, 3), dtype=dtype)
    actual = initialize_momenta(zero_pos, masses, torch.zeros(2, dtype=torch.long),
                                torch.tensor(1.0, dtype=dtype), generator=torch.Generator().manual_seed(4))
    raw = torch.randn((2, 3), dtype=dtype, generator=torch.Generator().manual_seed(4)) * masses.sqrt()[:, None]
    projected = raw - masses[:, None] * raw.sum(0) / masses.sum()
    native = Atoms("HHe", positions=np.zeros((2, 3)), masses=masses.numpy())
    native.set_momenta(raw.numpy())
    Stationary(native, preserve_temperature=False)
    ensembles = 32768
    sample = initialize_momenta(torch.zeros((2*ensembles, 3), dtype=dtype), masses.repeat(ensembles),
                               torch.arange(ensembles).repeat_interleave(2), torch.tensor(1.0, dtype=dtype),
                               generator=torch.Generator().manual_seed(71))
    findings["F005"] = {
        "status": "qualified_initialization_convention_and_distribution_finding",
        "scope": "unequal masses; both rules remove total momentum; JAX MD shares TorchSim rule",
        "masses": masses.tolist(), "kT": 1.0, "seed": 4, "raw_momenta": raw.tolist(),
        "torchsim_momenta": actual.tolist(), "ase_momenta": native.get_momenta().tolist(),
        "torchsim_total_momentum": actual.sum(0).tolist(),
        "ase_total_momentum": native.get_momenta().sum(0).tolist(),
        "ase_projection_error": float(np.max(np.abs(native.get_momenta()-projected.numpy()))),
        "max_projection_difference": float(np.max(np.abs(native.get_momenta()-actual.numpy()))),
        "mean_projection_variance": [1.25, 1.25],
        "mass_weighted_conditioned_gaussian_variance": [0.8, 0.8],
        "torchsim_ensemble_variance": sample.reshape(ensembles, 2, 3).var(dim=(0, 2), unbiased=True).tolist(),
        "ensemble_count": ensembles, "ensemble_seed": 71,
        "reference_invariant": "ASE Stationary(preserve_temperature=False) applies one common velocity shift to same raw momenta",
        "qualification": "not a failure to zero COM velocity and not unanimous cross-library evidence; distribution concern assumes canonical Gaussian momenta conditioned on zero total momentum",
    }

    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10]*3,
                  pbc=True, masses=[2.014, 2.014])
    imported = ts.io.atoms_to_state(atoms, dtype=dtype)
    exported = ts.io.state_to_atoms(imported)[0]
    with tempfile.TemporaryDirectory(prefix="torchsim-mass-control-") as folder:
        path = Path(folder) / "deuterium.traj"
        write(path, atoms, format="traj")
        native_masses = read(path, format="traj").get_masses().tolist()
    findings["F006"] = {
        "status": "observed_mass_loss_with_native_io_control",
        "scope": "explicit isotope/custom mass survives ASE trajectory IO but not TorchSim-to-ASE export",
        "input_masses": atoms.get_masses().tolist(), "state_masses": values(imported.masses),
        "exported_masses": exported.get_masses().tolist(),
        "native_ase_roundtrip_masses": native_masses,
        "ase_default_H_masses": Atoms("H2").get_masses().tolist(),
        "units": "amu", "reference_invariant": "explicit masses survive conversion round trip",
    }

    original = state([[0, 0, 0], [1, 0, 0]], [1, 1], dtype=torch.float32)
    single = ts.initialize_state(original, device=torch.device("cpu"), dtype=dtype)
    listed = ts.initialize_state([original, original], device=torch.device("cpu"), dtype=dtype)
    findings["F007"] = {
        "status": "observed_api_dtype_inconsistency",
        "scope": "API check, not physics comparison; CPU dtype tested; accelerator device conversion not executed",
        "input_dtype": str(original.dtype), "requested_dtype": str(dtype),
        "single_dtype": str(single.dtype), "list_dtype": str(listed.dtype),
        "requested_device": "cpu", "single_device": str(single.device), "list_device": str(listed.device),
        "reference_invariant": "same initialize_state dtype argument must apply to both SimState and list[SimState]",
        "external_physics_reference": "not applicable",
    }

    model = SiteSignalModel()
    mc = ts.swap_mc_init(state([[0, 0, 0], [1, 0, 0]], [1, 2]), model)
    before_positions, before_energy, before_signal = mc.positions.clone(), mc.energy.clone(), mc.site_signal.clone()
    proposal = mc.clone()
    proposal.positions = proposal.positions.flip(0)
    delta = float((model(proposal)["energy"] - before_energy).item())
    result = ts.swap_mc_step(mc, model, kT=1e-6, rng=torch.Generator().manual_seed(1))
    findings["F008"] = {
        "status": "observed_returned_state_extra_inconsistency",
        "scope": "state/API consistency check using deterministic toy model; not external physics comparison",
        "atomic_numbers": [1, 2], "original_positions": values(before_positions),
        "returned_positions": values(result.positions), "original_energy": values(before_energy),
        "returned_energy": values(result.energy), "original_signal": values(before_signal),
        "returned_signal": values(result.site_signal), "reevaluated_signal": values(model(result)["site_signal"]),
        "proposed_energy_delta": delta, "kT": 1e-6,
        "acceptance_probability": float(np.exp(-delta/1e-6)),
        "reference_invariant": "cached per-atom extras must match reevaluation of the returned, rejected state",
        "external_physics_reference": "not applicable",
    }
    ase_sources = {name: Path(inspect.getsourcefile(symbol)) for name, symbol in [
        ("ase/calculators/lj.py", LennardJones),
        ("ase/calculators/idealgas.py", IdealGas),
        ("ase/md/verlet.py", VelocityVerlet),
        ("ase/md/velocitydistribution.py", Stationary),
    ]}
    return {"schema_version": 1, "torchsim_commit": provenance["commit"],
            "source_provenance": provenance,
            "versions": {name: importlib.metadata.version(name) for name in ["torch-sim-atomistic", "torch", "ase", "numpy"]},
            "ase_source_file_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in ase_sources.items()},
            "backend": "CPU", "dtype": "float64 except F007 deliberate input float32",
            "success_semantics": "observations validated, not upstream fixes", "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/other-findings/results.json"))
    parser.add_argument("--jax-reference", type=Path, help="optional output from native_jaxmd.py in isolated JAX MD environment")
    parser.add_argument("--strict", action="store_true", help="exit 1 while these discrepancies remain (regression demonstration)")
    args = parser.parse_args()
    result = run()
    reference = json.loads(args.jax_reference.read_text()) if args.jax_reference else None
    ids = validate_evidence(result, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validated_observations": ids, "output": str(args.output),
                      "upstream_fixed": False, "strict_regression_exit": bool(args.strict)}))
    return int(args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
