"""Native TorchMD LJ reference. Run with this folder's isolated venv Python.

No force or energy formula is implemented here: the unmodified TorchMD
YamlForcefield -> Parameters -> Forces pipeline computes all reference values.
Both native epsilon=1 kcal/mol and physically epsilon=1 eV cases are emitted.
"""

from common import save, source_provenance
import torchmd
import tempfile
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import constants
import torch
from moleculekit.molecule import Molecule
from torchmd.forcefields.ff_yaml import YamlForcefield
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.systems import System


torch.set_default_dtype(torch.float64)
EV_TO_KCAL_MOL = constants.electron_volt * constants.Avogadro / (constants.kilo * constants.calorie)


def run_case(distance, epsilon, forcefield_path):
    mol = Molecule()
    mol.empty(2)
    mol.atomtype[:] = "AR"
    mol.element[:] = "Ar"
    mol.masses[:] = 1.0
    mol.charge[:] = 0.0
    forcefield_path.write_text(
        "atomtypes: [AR]\n"
        "lj:\n  AR:\n    sigma: 1.0\n"
        f"    epsilon: {epsilon:.17g}\n"
        "electrostatics:\n  AR:\n    charge: 0.0\n"
        "masses:\n  AR: 1.0\n"
    )
    ff = YamlForcefield(mol, str(forcefield_path))
    parameters = Parameters(ff, mol, terms=["lj"], precision=torch.float64, device="cpu")
    system = System(2, 1, torch.float64, "cpu")
    system.set_positions(torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]))
    # The zero box is the native documented code path for no periodic wrapping.
    system.set_box(np.zeros(3))
    model = Forces(parameters, terms=["lj"], cutoff=2.5, switch_dist=None, exclusions=())
    energy = model.compute(system.pos, system.box, system.forces, toNumpy=False)
    return {
        "distance": distance,
        "energy_kcal_mol": float(energy[0]),
        "forces_kcal_mol_angstrom": system.forces[0].tolist(),
        "forces": (system.forces[0] / epsilon).tolist(),
        "energy_reduced": float(energy[0] / epsilon),
        "unique_pair_indices": model.ava_idx.tolist(),
    }


def main():
    with tempfile.TemporaryDirectory(prefix="torchmd-native-") as directory:
        root = Path(directory)
        cases = [run_case(r, 1.0, root / "lj_reduced.yaml") for r in (1.1, 1.3, 1.5)]
        physical_cases = [run_case(r, EV_TO_KCAL_MOL, root / "lj_1ev.yaml") for r in (1.1, 1.3, 1.5)]
    for case in physical_cases:
        case["forces_ev_angstrom"] = (np.asarray(case["forces_kcal_mol_angstrom"]) / EV_TO_KCAL_MOL).tolist()
        case["energy_ev"] = case["energy_kcal_mol"] / EV_TO_KCAL_MOL
    data = {
        "engine": "TorchMD (torchmd/torchmd), not TorchMD-Net",
        "source_commit": source_provenance(torchmd.__file__, "torchmd", "09484183e34af78bb69b27a0149aeccf1318dbc7")["commit"],
        "version": importlib.metadata.version("torchmd"),
        "source_url": "https://github.com/torchmd/torchmd",
        "versions": {name: importlib.metadata.version(name) for name in ("torchmd", "torch", "moleculekit", "numpy", "scipy", "pyyaml")},
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
        "configuration": {"sigma_angstrom": 1.0, "epsilon_kcal_mol": 1.0, "cutoff_angstrom": 2.5, "switch_dist": None, "shift": False, "periodic": False, "positions": "[[0,0,0],[r,0,0]]", "terms": ["lj"]},
        "units": {"forces": "reduced force F*sigma/epsilon (dimensionless)", "native_energy": "kcal/mol", "native_force": "kcal/(mol Angstrom)", "length": "Angstrom"},
        "ev_to_kcal_mol": EV_TO_KCAL_MOL,
        "conversion_source": "scipy.constants: electron_volt * Avogadro / (kilo * calorie); calorie = thermochemical calorie = 4.184 J",
        "cases": cases,
        "physical_epsilon_1ev_cases": physical_cases,
        "native_call": "Forces(Parameters(YamlForcefield(mol, yaml), mol, terms=['lj'], precision=torch.float64, device='cpu'), terms=['lj'], cutoff=2.5, switch_dist=None, exclusions=()).compute(system.pos, system.box, system.forces, toNumpy=False)",
    }
    save(data)


if __name__ == "__main__":
    main()
