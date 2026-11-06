"""Record unmodified TorchSim forces; no expected force is hardcoded here."""

import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from common import save, source_provenance

import torch
import torch_sim as ts
from torch_sim.models.lennard_jones import (
    lennard_jones_pair,
    lennard_jones_pair_force,
)
from torch_sim.models.pair_potential import PairForcesModel, PairPotentialModel
from torch_sim.models.soft_sphere import soft_sphere_pair, soft_sphere_pair_force
from torch_sim.neighbors import torch_nl_n2


DTYPE = torch.float64
HERE = Path(__file__).resolve().parent


def state_at(distance):
    return ts.SimState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=DTYPE),
        masses=torch.ones(2, dtype=DTYPE),
        cell=10.0 * torch.eye(3, dtype=DTYPE).unsqueeze(0),
        pbc=False,
        atomic_numbers=torch.tensor([18, 18]),
    )


def lj_force(dr, zi, zj):
    # Only adapt the native helper's signature; do not change its expression.
    return lennard_jones_pair_force(dr, sigma=1.0, epsilon=1.0)


def record_case(distance, potential, force_fn, energy_fn, cutoff):
    state = state_at(distance)
    results = {}
    for backend, extra in [
        ("default", {}),
        ("torch_nl_n2", {"neighbor_list_fn": torch_nl_n2}),
    ]:
        common = dict(cutoff=cutoff, dtype=DTYPE, **extra)
        for label, model in [
            ("PairForcesModel_full_default", PairForcesModel(force_fn, **common)),
            (
                "PairForcesModel_half",
                PairForcesModel(force_fn, reduce_to_half_list=True, **common),
            ),
            ("PairPotentialModel", PairPotentialModel(energy_fn, **common)),
        ]:
            output = model(state.clone())
            results[f"{label}/{backend}"] = output["forces"].tolist()
    dr = torch.tensor([distance], dtype=DTYPE, requires_grad=True)
    zi = torch.tensor([18])
    energy = energy_fn(dr, zi, zi)
    derivative, = torch.autograd.grad(energy.sum(), dr)
    return {
        "potential": potential,
        "distance": distance,
        "positions": state.positions.tolist(),
        "epsilon": 1.0,
        "sigma": 1.0,
        "cutoff": cutoff,
        "pbc": False,
        "native_radial_force_helper": force_fn(dr.detach(), zi, zi).item(),
        "negative_energy_derivative": -derivative.item(),
        "unshifted_pair_energy": energy.item(),
        "forces": results,
    }


def main():
    provenance = source_provenance(ts.__file__, "torch-sim-atomistic", "d6d8e9fc30129e01a05b551c2d71912821c9b68b")
    result = {
        "engine": "TorchSim",
        "version": version("torch-sim-atomistic"),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "commit": provenance["commit"],
        "source_provenance": provenance,
        "source": "https://github.com/torchsim/torch-sim/tree/d6d8e9fc30129e01a05b551c2d71912821c9b68b",
        "dtype": "float64",
        "device": "cpu",
        "units": "epsilon=1 eV, sigma=1 angstrom; forces eV/angstrom",
        "cases": [
            record_case(r, "lennard_jones", lj_force, lennard_jones_pair, 2.5)
            for r in [1.1, 1.3, 1.5]
        ]
        + [record_case(0.5, "soft_sphere_alpha_2", soft_sphere_pair_force, soft_sphere_pair, 1.0)],
    }
    save(result)


if __name__ == "__main__":
    main()
