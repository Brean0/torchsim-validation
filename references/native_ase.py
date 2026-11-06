"""Native ASE LennardJones; adapted from the original executed audit script."""
import ase
from ase import Atoms
from ase.calculators.lj import LennardJones

from common import save


cases = []
for distance in (1.1, 1.3, 1.5):
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[20, 20, 20], pbc=False)
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5, smooth=False)
    cases.append({
        "distance": distance, "forces": atoms.get_forces().tolist(),
        "energy": float(atoms.get_potential_energy()),
        "per_particle_energy": atoms.get_potential_energies().tolist(),
    })
save({
    "engine": "ASE native ase.calculators.lj.LennardJones", "version": ase.__version__,
    "source": "https://gitlab.com/ase/ase/-/tree/3.29.0",
    "parameters": {"epsilon": 1.0, "sigma": 1.0, "rc": 2.5, "smooth": False},
    "units": {"distance": "Angstrom", "energy": "eV", "force": "eV/Angstrom"},
    "convention": "Native energy shifted at rc; force below rc unchanged. Energies are not cross-engine compared.",
    "cases": cases,
})
