"""Native LAMMPS shared-library lj/cut; no force formula in this adapter."""
from importlib.metadata import version
import lammps
from lammps import lammps as LAMMPS

from common import save


cases = []
for distance in (1.1, 1.3, 1.5):
    engine = LAMMPS(cmdargs=["-log", "none", "-screen", "none"])
    try:
        engine.commands_list([
            "units metal", "atom_style atomic", "boundary f f f",
            "atom_modify map array", "region box block -10 10 -10 10 -10 10 units box",
            "create_box 1 box", "create_atoms 1 single 0 0 0 units box",
            f"create_atoms 1 single {distance} 0 0 units box", "mass 1 39.948",
            "pair_style lj/cut 2.5", "pair_coeff 1 1 1.0 1.0 2.5",
            "pair_modify shift no tail no", "neighbor 0.3 bin",
            "neigh_modify every 1 delay 0 check yes", "compute atom_pe all pe/atom",
            "thermo_style custom step pe", "thermo_modify norm no", "run 0",
        ])
        forces = engine.gather_atoms("f", 1, 3)
        per_atom = engine.extract_compute("atom_pe", 1, 1)
        cases.append({
            "distance": distance,
            "forces": [[forces[3*i+j] for j in range(3)] for i in range(2)],
            "energy": float(engine.get_thermo("pe")),
            "per_particle_energy": [float(per_atom[i]) for i in range(2)],
        })
        native_version = engine.version()
    finally:
        engine.close()
save({
    "engine": "LAMMPS native shared library; built-in pair_style lj/cut",
    "version": native_version, "package_version": version("lammps"),
    "python_module_version": lammps.__version__, "mpi_package_version": version("mpich"),
    "source": "https://pypi.org/project/lammps/2025.7.22.4.0/",
    "parameters": {"epsilon": 1.0, "sigma": 1.0, "cutoff": 2.5, "shift": "no", "tail": "no"},
    "units": {"style": "metal", "distance": "Angstrom", "energy": "eV", "force": "eV/Angstrom"},
    "convention": "gather_atoms returns forces in atom ID order; native half list with Newton pair on; energy unshifted",
    "cases": cases,
})
