"""Independent native JAX MD reference for F003 and the F005 qualification.

Run in the isolated environment from requirements/jaxmd.txt. This script never
imports TorchSim and obtains energies/forces from native JAX MD APIs.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def installed_jaxmd_version():
    """Require the actual installed distribution pin before native execution."""
    actual = importlib.metadata.version("jax-md")
    if actual != "0.2.29":
        raise ValueError(f"JAX MD installed version is {actual}; expected 0.2.29")
    return actual


def run():
    actual_version = installed_jaxmd_version()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from jax_md import energy, quantity, simulate, space

    positions = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float64)
    sigma = jnp.array([[1.0, 2.0], [2.0, 2.0]], dtype=jnp.float64)
    displacement, _ = space.free()
    potential = energy.soft_sphere_pair(displacement, species=jnp.array([0, 0, 1]),
                                       sigma=sigma, epsilon=1.0,
                                       alpha=jnp.full((2, 2), 2.5, dtype=jnp.float64))
    native_energy = potential(positions)
    native_forces = quantity.force(potential)(positions)

    # Match native JAX MD's subkey path, not TorchSim's random number stream.
    # Comparing the same raw Gaussian sample within this engine establishes
    # which projection its public initializer actually uses.
    key = jax.random.PRNGKey(4)
    mass = jnp.array([[1.0], [4.0]], dtype=jnp.float64)
    pos = jnp.zeros((2, 3), dtype=jnp.float64)
    raw = jnp.sqrt(mass) * jax.random.normal(jax.random.split(key, 1)[0], pos.shape, dtype=pos.dtype)
    state = simulate.NVEState(position=pos, momentum=jnp.zeros_like(pos),
                              force=jnp.zeros_like(pos), mass=mass)
    initialized = simulate.initialize_momenta(state, key, 1.0)
    mean_projection = raw - jnp.mean(raw, axis=0, keepdims=True)
    velocity_projection = raw - mass * jnp.sum(raw, axis=0) / jnp.sum(mass)
    files = {name: Path(inspect.getsourcefile(module)) for name, module in
             [("jax_md/energy.py", energy), ("jax_md/simulate.py", simulate)]}
    return {
        "schema_version": 1, "engine": "JAX MD", "version": actual_version,
        "jax_version": jax.__version__, "jaxlib_version": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(), "dtype": "float64", "x64_enabled": bool(jax.config.jax_enable_x64),
        "source_distribution": "https://files.pythonhosted.org/packages/d6/2f/e4763555e3b8c66805090147956f57f0d134a67fb8aaccadcebec7362425/jax_md-0.2.29-py3-none-any.whl",
        "source_release": "https://pypi.org/project/jax-md/0.2.29/",
        "source_file_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        "fractional_soft_sphere": {
            "api": "jax_md.energy.soft_sphere_pair + jax_md.quantity.force",
            "positions": positions.tolist(), "species": [0, 0, 1], "sigma_matrix": sigma.tolist(),
            "epsilon": 1.0, "alpha": 2.5,
            "energy": float(native_energy), "forces": np.asarray(native_forces).tolist(),
            "scope": "same species-dependent sigma and fractional alpha; all pairs outside interaction support",
        },
        "unequal_mass_initialization": {
            "api": "jax_md.simulate.initialize_momenta", "masses": [1.0, 4.0], "kT": 1.0, "seed": 4,
            "raw_momenta": raw.tolist(), "initialized_momenta": initialized.momentum.tolist(),
            "total_momentum": jnp.sum(initialized.momentum, axis=0).tolist(),
            "mean_projection_max_error": float(jnp.max(jnp.abs(initialized.momentum - mean_projection))),
            "ase_style_projection_max_error": float(jnp.max(jnp.abs(initialized.momentum - velocity_projection))),
            "conclusion": "JAX MD shares TorchSim's mean-momentum projection; both remove COM momentum. This qualifies F005, not unanimous bug corroboration.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evidence/other-findings/jaxmd.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"engine": result["engine"], "version": result["version"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
