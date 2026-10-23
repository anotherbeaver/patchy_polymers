#!/bin/bash

LAMMPS=lmp
INPUTEQ=in.polymers_eq_Tg
INPUTPROD=in.polymers_prod_Tg
LOGDIR=logs
DATADIR=data
DUMPSDIR=data/dumps
AUTOCORRDIR=data/autocorrelation
PRESSDIR=data/pressures
RESTARTSDIR=data/restarts
PATCHESDIR=data/patches

mkdir -p "$LOGDIR" "$DATADIR" "$DUMPSDIR" "$AUTOCORRDIR" "$PRESSDIR" "$RESTARTSDIR" "$PATCHESDIR"

# -----------------------------
# Parameter space
# -----------------------------
N_list=(31)
N_beads_list=(24800)
patch_spacing_list=(5)
patch_strength_list=(0)
random_seed_list=(12345)

# -----------------------------
# Run Tg simulations
# -----------------------------
for index in "${!patch_strength_list[@]}"; do
  N=${N_list[0]}
  N_beads=${N_beads_list[0]}
  patch_spacing=${patch_spacing_list[0]}
  patch_strength=${patch_strength_list[index]}
  random_seed=${random_seed_list[0]}

  echo "Running Tg: N=${N}, N_beads=${N_beads}, patch_spacing=${patch_spacing}, patch_strength=${patch_strength}, random_seed=${random_seed}"

  export OMP_NUM_THREADS=4
  export OMP_PROC_BIND=close

  python3 generate_molecule.py \
    --chain_length ${N} \
    --patch_spacing ${patch_spacing} \
    --backbone_to_patch 0.25

  mpirun -n 8 ${LAMMPS} -in "${INPUTEQ}" \
    -sf omp \
    -pk omp ${OMP_NUM_THREADS} \
    -var chain_length "${N}" \
    -var N_beads "${N_beads}" \
    -var patch_spacing "${patch_spacing}" \
    -var patch_gauss_A_strength "${patch_strength}" \
    -var random_seed "${random_seed}"

  mpirun -n 8 ${LAMMPS} -in "${INPUTPROD}" \
    -sf omp \
    -pk omp ${OMP_NUM_THREADS} \
    -var chain_length "${N}" \
    -var N_beads "${N_beads}" \
    -var patch_spacing "${patch_spacing}" \
    -var patch_gauss_A_strength "${patch_strength}" \
    -var random_seed "${random_seed}"

  echo "Finished at $(date)"
done
