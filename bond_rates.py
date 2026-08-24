import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from time import time

cutoffs = [0.4, 0.5, 0.6, 0.7]
# As = [60, 70, 80, 90, 100, 110, 120, 130, 140]
As = [0]
spacing = 5

# in_dir = "data/patches"
# in_dir = "/mnt/hdd1/patchy_polymers_files"
out_dir = "data"

def biexponential(x, a1, b1, a2, b2):
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x)

def single_exponential(x, a, b):
    return a * np.exp(-b * x)

# given an association state s, we can convert it to a binary representation
def association_to_s(state):
    s = 0
    for i, val in enumerate(state):
        if val == 1:
            s += 2**i
    return s

import pandas as pd

def get_current_bonds(file_iter, N_patches, patch_molecule_map=None, pairs=False):
    current_bonds = set() # track current bonded pairs in this frame to compare with bonded_pairs dict
    clusters = pd.read_csv(file_iter, delimiter='\s+', header=None, nrows=N_patches, usecols=[0,1,5,6], names=['id', 'mol', 'patch_coord', 'patch_cluster'])
    bonded_clusters = clusters.loc[clusters['patch_coord'] > 0].copy() # filter only consider bonds with at least 1 patch (patch_coord is in column 5)
    bonded_clusters.sort_values(by='patch_cluster', inplace=True) # sort by patch_cluster (get bonded pairs) (patch_cluster is in column 6)
    bonded_clusters.reset_index(drop=True, inplace=True)

    arr = bonded_clusters.to_numpy()

    with open('temp.txt', "w") as f:
        for id in np.unique(clusters['id']):
            f.write(str(id) + "\n")

    for i in range(len(arr)-1):
        id1, id2 = int(arr[i, 0]), int(arr[i+1, 0]) # get ids of current and next cluster
        if int(arr[i, 2]) == int(arr[i+1, 2]): # if same patch_cluster, they are bonded
            if pairs:
                patch_molecule_map[id1] = int(arr[i, 1]) # map patch id to molecule id
                patch_molecule_map[id2] = int(arr[i+1, 1]) # map patch id to molecule id
                
                ids = (id1, id2) if id1 < id2 else (id2, id1) # order ids to avoid duplicates
                current_bonds.add(ids) # add to current bonds set
            else:
                current_bonds.add(id1) # add to current bonds set
                current_bonds.add(id2) # add to current bonds set
    
    return current_bonds

def count_s(bonded_pairs, patch_molecule_map, N_a=7, N_b=31, M=5, N_chains=800):
    bonded_patches = set() # track patches that are currently bonded
    molecule_ids = set() # track unique molecule ids that are currently bonded
    counts = {}

    for ids in bonded_pairs.keys():
        bonded_patches.add(ids[0])
        bonded_patches.add(ids[1])

        molecule_ids.add(patch_molecule_map[ids[0]])
        molecule_ids.add(patch_molecule_map[ids[1]])

    for mol_id in molecule_ids:
        state = [0] * N_a
        patches = [patch_id for patch_id, m_id in patch_molecule_map.items() if m_id == mol_id]

        for patch_id in patches:
            if patch_id in bonded_patches:
                state_id = patch_id - (N_b + N_a) * (mol_id - 1) - N_b
                state[state_id - 1] = 1

        s = association_to_s(state)
        counts[s] = counts.get(s, 0) + 1

    counts[0] = N_chains - len(molecule_ids)  # count unbonded chains (s=0)

    return counts

"""
Process number density of association state s from clustering data file. Each frame of the file contains the following sections:
- "ITEM: TIMESTEP" followed by the timestep on the next line
- "ITEM: NUMBER OF ATOMS" followed by the number of entries on the next line
- "ITEM: ATOMS" followed by the data lines for each entry, with 7 values per line: id, mol, x, y, z, patch_coord, patch_cluster

Args:
    - filename_on: path to the clustering data file for determining bond creation
    - filename_off: path to the clustering data file for determining bond destruction
    - frames: number of frames to read, starting from the beginning
Returns:
    - count_time_series: dict with (time, counts) pairs where counts is dict with all valid s and their counts at that time
"""
def process_buffered_s_counts(filename_on, filename_off, frames=1, timestep=0.002):
    # init empty bond lifetime list
    bond_lifetimes = []
    # init empty bonded pair dict. lookup by (id1, id2) where id1 < id2 to avoid duplicates. value is start_time of the bond
    bonded_pairs = {}
    patch_molecule_map = {} # map patch id to molecule id

    count_time_series = {}

    N_patches = 0
    time = 0
    read_frames = 0
    with open(filename_on, 'r') as f_on:
        with open(filename_off, 'r') as f_off:
            while True:
                # stop reading BOTH if either file is done
                # reading f_on, only creating bonds
                line = f_on.readline()
                if not line:
                    break
                if line.startswith("ITEM: TIMESTEP"): # timestep is on next line
                    time = float(f_on.readline().strip()) * timestep # convert to time units
                    # print(f"Processing timestep: {time / timestep:.0f} (time units: {time})")
                elif line.startswith("ITEM: NUMBER OF ATOMS"): # number of entries is on next line
                    N_patches = int(f_on.readline().strip())
                elif line.startswith("ITEM: ATOMS"): # data starts on next line
                    current_bonds = get_current_bonds(f_on, N_patches, patch_molecule_map, pairs=True)
                    
                    for ids in current_bonds:
                        if ids not in bonded_pairs: # if new bond, add to bonded_pairs dict with start_time as current timestep
                            bonded_pairs[ids] = time

                # reading f_off, only breaking bonds
                line = f_off.readline()
                if not line:
                    break
                if line.startswith("ITEM: TIMESTEP"): # timestep is on next line
                    time = float(f_off.readline().strip()) * timestep # convert to time units
                    # print(f"Processing timestep: {time / timestep:.0f} (time units: {time})")
                elif line.startswith("ITEM: NUMBER OF ATOMS"): # number of entries is on next line
                    N_patches = int(f_off.readline().strip())
                elif line.startswith("ITEM: ATOMS"): # data starts on next line
                    current_bonds = get_current_bonds(f_off, N_patches, patch_molecule_map, pairs=True)

                    broken_bonds = bonded_pairs.keys() - current_bonds # bonds that are in bonded_pairs but not in current_bonds are broken
                    for ids in broken_bonds:
                        bonded_pairs.pop(ids) # remove from bonded_pairs and get start_time

                    # after processing broken bonds, go calculate number of chains in association state s
                    counts = count_s(bonded_pairs, patch_molecule_map)

                    count_time_series[time] = counts
                    read_frames += 1
                    if read_frames >= frames:
                        break
                    
    return count_time_series



def process_and_save_s_counts(cutoff_on, cutoff_off, A):
    start = time()
    filename_on = f"data/patches/{cutoff_on}prodpatch_chainlength_31_patchspacing_5_N_beads_24800_gaussA_{A}_r0_0.25_gaussB_10_seed_12345.bin.txt"
    filename_off = f"data/patches/{cutoff_off}prodpatch_chainlength_31_patchspacing_5_N_beads_24800_gaussA_{A}_r0_0.25_gaussB_10_seed_12345.bin.txt"
    s_counts = process_buffered_s_counts(filename_on, filename_off, frames=10000)
    s_counts_df = pd.DataFrame.from_dict(s_counts, orient="index")
    s_counts_df.index.name = "time"
    s_counts_df = s_counts_df.fillna(0).astype(int)
    s_counts_df.to_csv(f"data/s_states/buffered_s_counts_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25.csv", index=False)
    print(f"Processed s_counts for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}, total s_counts recorded: {len(s_counts)} in {time() - start:.2f} seconds")




if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--out_dir', type=str, default=out_dir, help='Output directory')
    args.add_argument('--cutoffs', type=float, nargs='*', default=cutoffs, help='Cutoff distances to process')
    args.add_argument('--As', type=int, nargs='*', default=As, help='A values to process')
    args.add_argument('--spacing', type=int, default=spacing, help='Patch to patch spacing')
    args = args.parse_args()

    out_dir = args.out_dir
    cutoffs = args.cutoffs
    As = args.As
    spacing = args.spacing

    # tau_list = []

    # cutoff_on_list = [0.4, 0.5, 0.6, 0.7]
    # cutoff_off_list = [0.5, 0.6, 0.7]

    cutoff_on_list = cutoffs
    cutoff_off_list = cutoffs

    for A in As:
        for cutoff_off in cutoff_off_list:
            # fig = plt.figure(figsize=(8, 6))
            for cutoff_on in cutoff_on_list:
                if cutoff_on <= cutoff_off:
                    try:
                        df = pd.read_csv(f"{out_dir}/lifetimes/buffered_lifetimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv")
                        lifetimes = np.array(df["lifetime"].values)

                        # values, counts = np.unique(lifetimes, return_counts=True)
                        hist, bins = np.histogram(lifetimes, bins=50)
                        # print(f"Processed lifetimes for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}, total lifetimes recorded: {len(lifetimes)}")
                        hist = hist[1:] # remove first bin
                        bins = bins[1:] # remove first bin
                        # print(f"Removing, remaining lifetimes recorded: {np.sum(hist)}")
                        coeffs, cov = curve_fit(biexponential, bins[:-1], hist, p0=[1, 0.01, 1, 0.01], maxfev=100000000)
                        # print(f"Fitted coefficients for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}: {coeffs}")
                        tau_1, tau_2 = sorted((1/coeffs[1], 1/coeffs[3]))
                        # print(f"Fitted lifetimes for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}: tau_1={tau_1:.2f}, tau_2={tau_2:.2f}")

                        tau_list = [A, "life", cutoff_on, cutoff_off, tau_1, tau_2]
                        df = pd.DataFrame([tau_list], columns=["A", "off/life", "cutoff_on", "cutoff_off", "tau_1", "tau_2"])
                        df.to_csv(f"{out_dir}/lifetimes/tau_lifetimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv", index=False)

                        print("saving to binned")
                        df = pd.DataFrame([hist, bins[:-1]], index=["hist", "bins"])
                        df.to_csv(f"{out_dir}/lifetimes/binned/binned_lifetimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv", index=False)

                        print(f"Finished saving binned lifetimes to {out_dir}/lifetimes/binned/binned_lifetimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv")
                        # plt.plot(bins[:-1], hist, alpha=0.5, label=f"A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}")
                        # plt.plot(bins[:-1], biexponential(bins[:-1], *coeffs), '--', label=r"Fit $\tau_1=$" + f"{tau_1:.2f}" + r", $\tau_2=$" + f"{tau_2:.2f}, lifetimes")

                    except:
                        print(f"Failed to fit biexponential for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}")
                        tau_1, tau_2 = np.nan, np.nan

                    
                    
            # plt.xlabel("Lifetime (LJ units)")
            # plt.ylabel("Density")
            # plt.yscale("log")
            # # plt.ylim(0, 0.2)
            # plt.grid()
            # plt.title(f"Distribution of Bond Lifetimes Assorted r_off, r_on={cutoff_on} for A={A}")
            # plt.legend()
            # plt.show()


    
    # tau_list = []


    for A in As:
        for cutoff_on in cutoff_on_list:
            # fig = plt.figure(figsize=(8, 6))
            for cutoff_off in cutoff_off_list:
                if cutoff_on <= cutoff_off:
                    try:
                        df = pd.read_csv(f"{out_dir}/offtimes/buffered_offtimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv")
                        offtimes = np.array(df["offtime"].values)

                        # values, counts = np.unique(lifetimes, return_counts=True)
                        hist, bins = np.histogram(offtimes, bins=50)
                        # print(f"Processed offtimes for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}, total lifetimes recorded: {len(offtimes)}")
                        hist = hist[1:] # remove first bin
                        bins = bins[1:] # remove first bin
                        # print(f"Removing, remaining offtimes recorded: {np.sum(hist)}")
                        coeffs, cov = curve_fit(biexponential, bins[:-1], hist, p0=[1, 0.01, 1, 0.01], maxfev=100000000)
                        # print(f"Fitted coefficients for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}: {coeffs}")
                        tau_1, tau_2 = sorted((1/coeffs[1], 1/coeffs[3]))
                        # print(f"Fitted offtimes for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}: tau_1={tau_1:.2f}, tau_2={tau_2:.2f}")
                        tau_list = [A, "off", cutoff_on, cutoff_off, tau_1, tau_2]
                        df = pd.DataFrame([tau_list], columns=["A", "off/life", "cutoff_on", "cutoff_off", "tau_1", "tau_2"])
                        df.to_csv(f"{out_dir}/offtimes/tau_offtimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv", index=False)

                        df = pd.DataFrame([hist, bins[:-1]], index=["hist", "bins"])
                        df.to_csv(f"{out_dir}/offtimes/binned/binned_offtimes_A{A}_cutoff_on{cutoff_on}_cutoff_off{cutoff_off}_r0_0.25_spacing_{spacing}.csv", index=False)

                        # plt.plot(bins[:-1], hist, alpha=0.5, label=f"A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}")
                        # plt.plot(bins[:-1], biexponential(bins[:-1], *coeffs), '--', label=r"Fit $\tau_1=$" + f"{tau_1:.2f}" + r", $\tau_2=$" + f"{tau_2:.2f}, offtimes")

                    except:
                        print(f"Failed to fit biexponential for A={A}, cutoff_on={cutoff_on}, cutoff_off={cutoff_off}")
                        tau_1, tau_2 = np.nan, np.nan

                    

            # plt.xlabel("Lifetime (LJ units)")
            # plt.ylabel("Density")
            # plt.yscale("log")
            # # plt.ylim(0, 0.2)
            # plt.grid()
            # plt.title(f"Distribution of Bond Offtimes Assorted r_on, r_off={cutoff_off} for A={A}")
            # plt.legend()
            # plt.show()




    # life_tau_df = pd.read_csv(f"{out_dir}/lifetimes/tau_lifetimes_spacing_5.csv")
    # off_tau_df = pd.read_csv(f"{out_dir}/offtimes/tau_offtimes_spacing_5.csv")

    # # plot tau_2 for life over tau_2 for off for each A, cutoff_on, cutoff_off
    # plt.figure(figsize=(8, 6))
    # As = [0]

    # for A in As:
    #     for cutoff_off in cutoff_off_list:
    #         tau_life_over_off_list = []
    #         saved_cutoff_on_list = []
    #         for cutoff_on in cutoff_on_list:
    #             if cutoff_on <= cutoff_off:
    #                 tau_life = life_tau_df.loc[(life_tau_df["A"] == A) & (life_tau_df["off/life"] == "life") & (life_tau_df["cutoff_on"] == cutoff_on) & (life_tau_df["cutoff_off"] == cutoff_off), "tau_2"].values[0]
    #                 tau_off = off_tau_df.loc[(off_tau_df["A"] == A) & (off_tau_df["off/life"] == "off") & (off_tau_df["cutoff_on"] == cutoff_on) & (off_tau_df["cutoff_off"] == cutoff_off), "tau_2"].values[0]
    #                 tau_life_over_off_list.append(tau_life / tau_off)
    #                 print(tau_life_over_off_list)
    #                 saved_cutoff_on_list.append(cutoff_on)

    #         plt.plot(saved_cutoff_on_list, np.array(tau_life_over_off_list), label=f"A={A}, cutoff_on={cutoff_on}", alpha=0.7)
    #         plt.xlabel(r"$r_{\text{on}}$ for (LJ distance units)")
    #         plt.ylabel(r"$\tau_{life}/\tau_{off}$ (LJ time units)")
    #         plt.title(f"Comparison of \\alpha/\\beta for Bond Lifetimes and Offtimes, cutoff_off={cutoff_off}")
    #         # plt.xscale("log")
    #         # plt.yscale("log")
    #         plt.grid()
    #         plt.legend()
    #         plt.show()
    # only process lifetimes for cutoff_on < cutoff_off to avoid duplicates
    # finish = Parallel(n_jobs=10)(delayed(process_and_save_s_counts)(cutoff_on, cutoff_off, A) for cutoff_on in cutoffs for cutoff_off in cutoffs if cutoff_on <= cutoff_off for A in As)



