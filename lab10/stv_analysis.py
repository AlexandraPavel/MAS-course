from preflibtools import io
from preflibtools.generate_profiles import gen_mallows, gen_cand_map, gen_impartial_culture_strict
from typing import List, Dict, Tuple
import random
import math
import copy
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

PHIS = [0.7, 0.8, 0.9, 1.0]
NUM_VOTERS = [100, 500, 1000]
NUM_CANDIDATES = [3, 6, 10, 15]


def generate_random_mixture(nvoters: int = 100, ncandidates: int = 6, num_refs: int = 3, phi: float = 1.0) \
    -> Tuple[Dict[int, str], List[Dict[int, int]], List[int]]:
    """
    Function that will generate a `voting profile` where there are num_refs mixtures of a
    Mallows model, each with the same phi hyperparameter
    :param nvoters: number of voters
    :param ncandidates: number of candidates
    :param num_refs: number of Mallows Mixtures in the voting profile
    :param phi: hyper-parameter for each individual Mallows model
    :return: a tuple consisting of:
        the candidate map (map from candidate id to candidate name),
        a ranking list (list consisting of dictionaries that map from candidate id to order of preference)
        a ranking count (the number of times each vote order comes up in the ranking list,
        i.e. one or more voters may end up having the same preference over candidates)
    """
    candidate_map = gen_cand_map(ncandidates)

    mix = []
    phis = []
    refs = []

    for i in range(num_refs):
        refm, refc = gen_impartial_culture_strict(1, candidate_map)
        refs.append(io.rankmap_to_order(refm[0]))
        phis.append(phi)
        mix.append(random.randint(1,100))

    smix = sum(mix)
    mix = [float(m)/float(smix) for m in mix]

    rmaps, rmapscounts = gen_mallows(nvoters, candidate_map, mix, phis, refs)

    return candidate_map, rmaps, rmapscounts


def stv(nvoters: int,
        candidate_map: Dict[int, str],
        rankings: List[Dict[int, int]],
        ranking_counts: List[int],
        top_k: int,
        required_elected: int) -> List[int]:
    """
    :param nvoters: number of voters
    :param candidate_map: the mapping of candidate IDs to candidate names
    :param rankings: the expressed full rankings of voters, specified as a list of mapping from candidate_id -> rank
    :param ranking_counts: the number of times each ranking was cast
    :param top_k: the number of preferences taken into account [min: 2, max: (num_candidates - 1), aka full STV]
    :param required_elected: the number of candidates to be elected
    :return: The list of elected candidate id-s
    """
    candidates = set(candidate_map.keys())
    elected = []
    remaining_candidates = set(candidates)

    def redistribute_surplus_votes(tally, elected_candidate, surplus):
        next_preferences = defaultdict(int)

        for ranking, count in zip(rankings, ranking_counts):
            if elected_candidate in ranking:
                rank = ranking[elected_candidate]
                for candidate, candidate_rank in ranking.items():
                    if candidate_rank == rank + 1 and candidate in remaining_candidates:
                        next_preferences[candidate] += count

        total_next_preferences = sum(next_preferences.values())

        if total_next_preferences > 0:
            for candidate in next_preferences:
                transfer_votes = (next_preferences[candidate] / total_next_preferences) * surplus
                tally[candidate] += transfer_votes

        return tally

    surplus = 0
    last_cadidated_elected = None

    while len(elected) < required_elected and remaining_candidates:
        tally = {candidate: 0 for candidate in remaining_candidates}
        if last_cadidated_elected:
            redistribute_surplus_votes(tally, last_cadidated_elected, surplus)

        for ranking, count in zip(rankings, ranking_counts):
            for rank in range(1, top_k + 1):
                for candidate, candidate_rank in ranking.items():
                    if candidate_rank == rank and candidate in remaining_candidates:
                        tally[candidate] += count
                        break

        min_votes = min(tally.values())
        candidates_with_min_votes = [c for c in remaining_candidates if tally[c] == min_votes]

        if len(candidates_with_min_votes) == len(remaining_candidates):
            random.shuffle(candidates_with_min_votes)

        candidate_to_eliminate = candidates_with_min_votes[0]
        remaining_candidates.remove(candidate_to_eliminate)

        if len(remaining_candidates) == required_elected - len(elected):
            elected.extend(remaining_candidates)
            break

        quota = nvoters // (required_elected + 1) + 1
        for candidate, votes in tally.items():
            if votes >= quota and candidate in remaining_candidates:
                surplus = votes - quota
                last_cadidated_elected = candidate
                elected.append(candidate)
                remaining_candidates.remove(candidate)
                if len(elected) == required_elected:
                    break

    return elected


def run_experiment(num_experiments=1000):
    results = defaultdict(lambda:  defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
        

    # for num_voters in NUM_VOTERS:
    #     for num_candidates in NUM_CANDIDATES:
    #         for required_elected in [2, num_candidates // 2, num_candidates-1]:
    #             for phi in PHIS:
    #                 for k in range(2, num_candidates):
    #                     overlaps = []
    #                     print("num_voters", num_voters, "num_candidates", num_candidates, "phi", phi, "top_k", k)
    #                     for _ in range(num_experiments):
    #                         cmap, rmaps, rmapscounts = generate_random_mixture(
    #                             nvoters = num_voters,\
    #                             ncandidates=num_candidates,\
    #                             phi=phi)
                            
    #                         stv_winner = stv(num_voters, cmap, rmaps, rmapscounts, num_candidates - 1, required_elected)
    #                         stv_k_winner = stv(num_voters, cmap, rmaps, rmapscounts, k, required_elected)
    #                         # overlap = compute_overlap(stv_winner, stv_k_winner)
    #                         # Compute the overlap between STV and STV-k winners
    #                         overlap = [stv_k_winner == stv_winner]
    #                         # print(overlap)
    #                         overlaps.append(overlap)
    #                     average_overlap = np.mean(overlaps)
    #                     print(average_overlap)
    #                     results[num_voters][num_candidates][required_elected][phi][k] = average_overlap
        #                 break
        #             break
        #         break
        #     break
        # break

    # For 100 voters and required elected mid number
    # for num_voters in [100]:
    #     for num_candidates in NUM_CANDIDATES:
    #         for required_elected in [num_candidates - 1]:
    #             for phi in PHIS:
    #                 for k in range(2, num_candidates):
    #                     overlaps = []
    #                     print("num_voters", num_voters, "num_candidates", num_candidates, "phi", phi, "top_k", k)
    #                     for _ in range(num_experiments):
    #                         cmap, rmaps, rmapscounts = generate_random_mixture(
    #                             nvoters = num_voters,\
    #                             ncandidates=num_candidates,\
    #                             phi=phi)
                            
    #                         stv_winner = stv(num_voters, cmap, rmaps, rmapscounts, num_candidates - 1, required_elected)
    #                         stv_k_winner = stv(num_voters, cmap, rmaps, rmapscounts, k, required_elected)
    #                         # overlap = compute_overlap(stv_winner, stv_k_winner)
    #                         # Compute the overlap between STV and STV-k winners
    #                         overlap = [stv_k_winner == stv_winner]
    #                         # print(overlap)
    #                         overlaps.append(overlap)
    #                     average_overlap = np.mean(overlaps)
    #                     print(average_overlap)
    #                     results[num_voters][num_candidates][required_elected][phi][k] = average_overlap

    # for num_voters in [500]:
    #     for num_candidates in NUM_CANDIDATES:
    #         for required_elected in [num_candidates - 1]:
    #             for phi in PHIS:
    #                 for k in range(2, num_candidates):
    #                     overlaps = []
    #                     print("num_voters", num_voters, "num_candidates", num_candidates, "phi", phi, "top_k", k)
    #                     for _ in range(num_experiments):
    #                         cmap, rmaps, rmapscounts = generate_random_mixture(
    #                             nvoters = num_voters,\
    #                             ncandidates=num_candidates,\
    #                             phi=phi)
                            
    #                         stv_winner = stv(num_voters, cmap, rmaps, rmapscounts, num_candidates - 1, required_elected)
    #                         stv_k_winner = stv(num_voters, cmap, rmaps, rmapscounts, k, required_elected)
    #                         # overlap = compute_overlap(stv_winner, stv_k_winner)
    #                         # Compute the overlap between STV and STV-k winners
    #                         overlap = [stv_k_winner == stv_winner]
    #                         # print(overlap)
    #                         overlaps.append(overlap)
    #                     average_overlap = np.mean(overlaps)
    #                     print(average_overlap)
    #                     results[num_voters][num_candidates][required_elected][phi][k] = average_overlap

    for num_voters in [1000]:
        for num_candidates in NUM_CANDIDATES:
            for required_elected in [num_candidates - 1]:
                for phi in PHIS:
                    for k in range(2, num_candidates):
                        overlaps = []
                        print("num_voters", num_voters, "num_candidates", num_candidates, "phi", phi, "top_k", k)
                        for _ in range(num_experiments):
                            cmap, rmaps, rmapscounts = generate_random_mixture(
                                nvoters = num_voters,\
                                ncandidates=num_candidates,\
                                phi=phi)
                            
                            stv_winner = stv(num_voters, cmap, rmaps, rmapscounts, num_candidates - 1, required_elected)
                            stv_k_winner = stv(num_voters, cmap, rmaps, rmapscounts, k, required_elected)
                            # overlap = compute_overlap(stv_winner, stv_k_winner)
                            # Compute the overlap between STV and STV-k winners
                            overlap = [stv_k_winner == stv_winner]
                            # print(overlap)
                            overlaps.append(overlap)
                        average_overlap = np.mean(overlaps)
                        print(average_overlap)
                        results[num_voters][num_candidates][required_elected][phi][k] = average_overlap

    with open('result_1000_voters_elected_n-1.json', 'w') as fp:
        json.dump(results, fp, indent=4)

    return results

def plot_results(results):

    for num_voters in NUM_VOTERS:
        for num_candidates in NUM_CANDIDATES:
            plt.figure(figsize=(20, 10))
            for required_elected in [2, num_candidates / 2, num_candidates - 1]:
                for phi in PHIS:
                    # for k in 
                    x = list(results[str(num_voters)][str(num_candidates)][str(required_elected)][str(phi)].keys())
                    y = [results[str(num_voters)][str(num_candidates)][str(required_elected)][str(phi)][k] for k in x]
                    plt.plot(x, y, marker='x', label=f" voters: {num_voters}, candidates: {num_candidates}, phi: {phi}, required_elected: {required_elected}")
                    

                plt.ylabel("Average Overlap")
                plt.xlabel("Top-k")
                plt.legend(loc='upper right',  bbox_to_anchor=(1.1, 1))
                plt.title("PHI variation")
                plt.savefig(f"phi/voters_{num_voters}_candidates_{num_candidates}_required_elected_{required_elected}.png")
                plt.close()

    for num_voters in NUM_VOTERS:
        for phi in PHIS:
            plt.figure(figsize=(20, 10))
            for num_candidates in NUM_CANDIDATES:
                for required_elected in [2, num_candidates / 2, num_candidates - 1]:
                # for k in 
                    x = list(results[str(num_voters)][str(num_candidates)][str(required_elected)][str(phi)].keys())
                    y = [results[str(num_voters)][str(num_candidates)][str(required_elected)][str(phi)][k] for k in x]
                    plt.plot(x, y, marker='x', label=f" voters: {num_voters}, phi: {phi}, candidates: {num_candidates}, required_elected: {required_elected}")

            plt.ylabel("Average Overlap")
            plt.xlabel("Top-k")
            plt.legend(loc='upper right',  bbox_to_anchor=(1.1, 1))
            plt.title("Num Candidates variation")
            plt.savefig(f"candidates/voters_{num_voters}_phi_{phi}.png")
            plt.close()

    for num_candidates in NUM_CANDIDATES:
        for phi in PHIS: 
            plt.figure(figsize=(20, 10))
            for required_elected in [2, num_candidates / 2, num_candidates - 1]:
                for num_voters in NUM_VOTERS:
                    # for k in 
                    x = list(results[str(num_voters)][str(num_candidates)][str(required_elected)][str(phi)].keys())
                    y = [results[str(num_voters)][str(num_candidates)][str(required_elected)][str(phi)][k] for k in x]
                    plt.plot(x, y, marker='x', label=f" candidates: {num_candidates}, phi: {phi}, voters: {num_voters}, required_elected: {required_elected}")
                    
                plt.ylabel("Average Overlap")
                plt.xlabel("Top-k")
                plt.legend(loc='upper right',  bbox_to_anchor=(1.1, 1))
                plt.title("Num Voters variation")
                plt.savefig(f"voters/candidates_{num_candidates}_phi_{phi}_required_elected_{required_elected}.png")
                plt.close()


if __name__ == "__main__":
    # cmap, rmaps, rmapscounts = generate_random_mixture()
    # print("cmap", cmap)
    # print("rmaps", rmaps)
    # print("rmapscounts", rmapscounts)

    # nvoters = len(rmaps)
    # top_k = 5  
    # required_elected = 3
    # result_k = stv(nvoters, cmap, rmaps, rmapscounts, top_k, required_elected)
    # result = stv(nvoters, cmap, rmaps, rmapscounts, top_k=len(cmap) - 1, required_elected=required_elected)
    # print("Result_k", result_k)
    # print("Result", result)

    # Run the experiment
    # results = run_experiment()

    # files = [
    #     'result_100_voters_elected_n-1.json',\
    #     'result_500_voters_elected_n-1.json',\
    #     'result_1000_voters_elected_n-1.json']

    # # If you want to merge dictionaries instead of lists, you can use:
    # combined_data = {}
    # for file in files:
    #     with open(file, 'r') as f:
    #         data = json.load(f)
    #         combined_data.update(data)

    # # Write the combined data to a new JSON file
    # with open('combined_3.json', 'w') as f:
    #     json.dump(combined_data, f, indent=4)


    # with open('combined.json', 'r') as f1, open('combined_2.json', 'r') as f2, open('combined_3.json', 'r') as f3:
    #     data1 = json.load(f1)
    #     data2 = json.load(f2)
    #     data3 = json.load(f3)

    # # Extract the relevant data based on the specified key
    # # key = "100"
    # data_file = [data2, data1, data3]
    # combined_data = {}
    # # required_elected = ["2", "num_candidates/2", "num_candidates - 1"]
    
    # for key in ["100", "500", "1000"]:
    #     combined_data[key] = {}
    #     for num_candidates in NUM_CANDIDATES:
    #         combined_data[key][num_candidates] = {}
    #         # for required_elected in [2, num_candidates / 2, num_candidates-1]:
    #         #     combined_data[key][num_candidates][required_elected] = {}
    #             # for i in ["2", "num_candidates/2", "num_candidates - 1"]:
    #             #     combined_data[key][i] = {}
    #         for i in range(len(data_file)):
    #             if key in data_file[i]:
    #                 # add = data_file[key]
    #                 # print(key, num_candidates, required_elected)
    #                 # print(data_file[i][key])
    #                 combined_data[key][num_candidates].update(data_file[i][key][str(num_candidates)])

    # # Write the combined data to a new JSON file
    # with open('all_results.json', 'w') as f:
    #     json.dump(combined_data, f, indent=4)

    # print("JSON files have been concatenated and saved to combined.json")

    with open("all_results.json", "r") as f:
        results = json.load(f)

    # print("Result", results)

    # new_results = {}
    # idx = 1
    # for num_voters in results:
    #     for num_candidates in results[num_voters]:
    #         for phi in results[num_voters][num_candidates]:
    #             new_results[idx] = {
    #                 "num_voters": num_voters,
    #                 "num_candidates": num_candidates,
    #                 "phi": phi,
    #                 "k_list": results[num_voters][num_candidates][phi]
    #             }

    #             idx += 1

    # print(new_results)

    # Plot the results
    plot_results(results)





