A_list=(0 60 70 80 90 100 110 120 130 140 150)
# A_list=(140 150)
# A_list=(0)



cutoff_list=(0.4 0.5 0.6 0.7)
spacing=5
# cutoff_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)

# set -e

python3 generate_molecule.py \
    --chain_length 31 \
    --patch_spacing "${spacing}" \
    --backbone_to_patch 0.25

for A in "${A_list[@]}"; do
    echo "Simulating A=${A}"
    mpirun -np 6 ../lammps/build/lmp \
        -in "in.polymers_lifetime_eq" \
        -var chain_length "31" \
        -var N_beads "24800" \
        -var patch_spacing "${spacing}" \
        -var patch_gauss_A_strength "$A" \
        -var random_seed "12345"

    wait

    mpirun -np 6 ../lammps/build/lmp \
        -in "in.polymers_lifetime_prod" \
        -var chain_length "31" \
        -var N_beads "24800" \
        -var patch_spacing "${spacing}" \
        -var patch_gauss_A_strength "$A" \
        -var random_seed "12345"

    wait


    for cutoff in "${cutoff_list[@]}"; do
        echo "postprocessing A=${A}, cutoff=${cutoff}"

        ../lammps/tools/binary2txt \
            "data/patches/${cutoff}prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin" &
    done

    wait

    # # move processed files to hdd1 (2 TB)
    # mv  "data/patches/0.4prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin.txt" \
    #     "data/patches/0.5prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin.txt" \
    #     "data/patches/0.6prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin.txt" \
    #     "data/patches/0.7prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin.txt" \
    #     "/mnt/hdd1/patchy_polymers_files"
    # wait


    for cutoff in "${cutoff_list[@]}"; do
        echo "removing binary file for A=${A}, cutoff=${cutoff}"
        rm "data/patches/${cutoff}prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin"
    done

    wait

    temp_A_list=("$A")

    python3 bond_off_lifetime.py \
    --spacing "${spacing}" \
    --As "${temp_A_list[@]}" \
    --cutoffs "${cutoff_list[@]}"

    python3 bond_rates.py \
        --spacing "${spacing}" \
        --As "${temp_A_list[@]}" \
        --cutoffs "${cutoff_list[@]}"

    # for cutoff in "${cutoff_list[@]}"; do
    #     echo "removing processed text file for A=${A}, cutoff=${cutoff}"
    #     rm "data/patches/${cutoff}prodpatch_chainlength_31_patchspacing_${spacing}_N_beads_24800_gaussA_${A}_r0_0.25_gaussB_10_seed_12345.bin.txt"
    # done


done

