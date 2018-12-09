#!/bin/bash

#SBATCH --image=docker:jedbrown/fenics:latest
#SBATCH --nodes=2
#SBATCH --qos=debug
#SBATCH -C haswell

srun -n 8 shifter --env=PETSC_OPTIONS='-pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_detect_saddle_point -pc_fieldsplit_schur_fact_type lower -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_ksp_monitor -fieldsplit_1_ksp_monitor -ksp_monitor -fieldsplit_0_ksp_type preonly -fieldsplit_0_sub_pc_type icc -fieldsplit_0_pc_factor_mat_ordering_type rcm -fieldsplit_1_ksp_converged_reason -fieldsplit_1_pc_type hypre -fieldsplit_1_ksp_type preonly -log_view' python3 solver.py
