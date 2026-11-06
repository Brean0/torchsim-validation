"""Native JAX MD reference for the TorchSim pair-force duplication audit.

Run with this directory's isolated .venv/bin/python. No TorchSim imports,
custom pair potentials, custom integration, or modifications of JAX MD.
"""
from common import save
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md import energy, quantity, space


def main():
    displacement, _ = space.free()
    total_energy = energy.lennard_jones_pair(
        displacement, sigma=1.0, epsilon=1.0, r_onset=2.0, r_cutoff=2.5
    )
    per_particle_energy = energy.lennard_jones_pair(
        displacement, sigma=1.0, epsilon=1.0, r_onset=2.0,
        r_cutoff=2.5, per_particle=True,
    )
    force = quantity.force(total_energy)
    cases = []
    for distance in [1.1, 1.3, 1.5]:
        positions = jnp.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
                              dtype=jnp.float64)
        forces = force(positions)
        assert forces.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(forces)))
        cases.append({
            "distance": distance,
            "positions": positions.tolist(),
            "energy": float(total_energy(positions)),
            "forces": forces.tolist(),
            "per_particle_energy": per_particle_energy(positions).tolist(),
            "per_particle_sum": float(per_particle_energy(positions).sum()),
        })
    # This is the original first-bug potential, evaluated by native JAX MD.
    soft_energy = energy.soft_sphere_pair(
        displacement, sigma=1.0, epsilon=1.0, alpha=2.0
    )
    soft_positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                               dtype=jnp.float64)
    module_path = Path(inspect.getsourcefile(energy))
    result = {
        "engine": "JAX MD",
        "version": importlib.metadata.version("jax-md"),
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "devices": [str(x) for x in jax.devices()],
        "dtype": "float64",
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "source": "https://github.com/jax-md/jax-md",
        "source_release": "https://pypi.org/project/jax-md/0.2.29/",
        "source_distribution": "https://files.pythonhosted.org/packages/d6/2f/e4763555e3b8c66805090147956f57f0d134a67fb8aaccadcebec7362425/jax_md-0.2.29-py3-none-any.whl",
        "source_distribution_sha256_from_pypi": "b4e27fe280458bc862f32c5343f47c946be68ccb863e0b3feb310e0a8524c92c",
        "primary_docs": [
            "https://jax-md.readthedocs.io/en/main/jax_md.energy.html#jax_md.energy.lennard_jones_pair",
            "https://jax-md.readthedocs.io/en/main/jax_md.quantity.html#jax_md.quantity.force",
            "https://jax-md.readthedocs.io/en/main/jax_md.space.html#jax_md.space.free",
        ],
        "source_file": "jax_md/energy.py",
        "source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "api": "jax_md.energy.lennard_jones_pair + jax_md.quantity.force",
        "potential": "native Lennard-Jones 12-6",
        "settings": {
            "sigma": 1.0, "epsilon": 1.0, "pbc": False,
            "r_onset": 2.0, "r_cutoff": 2.5,
            "switching": "JAX MD native multiplicative isotropic cutoff; all test distances are below onset, so switching is identically one",
            "energy_shift": "none",
            "force_method": "native quantity.force, negative JAX autodiff gradient of native total energy",
            "neighbor_list": "none: native all-pairs convenience API; exactly two particles and one unique pair",
        },
        "cases": cases,
        "original_soft_sphere_control": {
            "api": "jax_md.energy.soft_sphere_pair + jax_md.quantity.force",
            "sigma": 1.0, "epsilon": 1.0, "alpha": 2.0,
            "distance": 0.5,
            "positions": soft_positions.tolist(),
            "energy": float(soft_energy(soft_positions)),
            "forces": quantity.force(soft_energy)(soft_positions).tolist(),
        },
    }
    save(result)


if __name__ == "__main__":
    main()
