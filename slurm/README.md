## SLURM plugin for NVIDIA GPU power capping

This folder contains implementations of a SLURM plugin to enable GPU power capping in a SLURM job. The implementation consists of the following 

* `gpupower.c` - SLURM plugin written in C to enable GPU power capping 
* `slurm.prolog` - Job prolog to be added to the SLURM configuration. The prolog is responsible for using the `nvidia-smi` utility to set maximum GPU power.
* `slurm.epilog` - Job epilog to be added to the SLURM configuration. The epilog resets the GPU powert cap to the default value chosen by the system operator.

For details on SLURM plugins please see the following documentation : 
* https://slurm.schedmd.com/plugins.html
* https://slurm.schedmd.com/add.html
