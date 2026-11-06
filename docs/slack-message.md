# Slack message draft

Sharing five TorchSim bugs I reproduced and checked against independent native reference calculations. I've prepared a separate fix and regression tests for each, with the inputs, measured results and methodology in the validation repo:

https://github.com/Brean0/torchsim-validation

The findings cover doubled pair forces, per-atom energies that sum to twice the total, nonfinite soft-sphere forces outside a species cutoff, per-system timesteps applied to coordinate axes, and custom masses lost during ASE export.

The strongest comparison is the force case: ASE, LAMMPS, JAX MD and TorchMD agree, while one TorchSim direct-force option returns twice the force. Two other TorchSim options agree with the references. The other four findings have case-specific ASE or JAX MD controls; they weren't each compared against every library.

I've also documented three additional findings separately: two reproduced API/state consistency issues and one qualified initialization difference. The initialization case needs discussion of the intended convention because JAX MD shares TorchSim's rule while ASE uses a different one.

The repo includes a one-page overview and the proposed changes. I'd appreciate a maintainer review of the findings and fixes.
