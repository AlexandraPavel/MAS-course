import math

def stv(cmap, rmaps, rmapscounts, seats=3):
    # Count the first preferences
    first_prefs = {candidate: 0 for candidate in cmap.keys()}
    for rmap, count in zip(rmaps, rmapscounts):
        first_choice = min(rmap, key=rmap.get)
        first_prefs[first_choice] += count

    # Calculate the quota
    total_votes = sum(rmapscounts)
    quota = math.floor(total_votes / (seats + 1)) + 1

    elected = []
    while len(elected) < seats:
        # Check if any candidate meets the quota
        for candidate, votes in first_prefs.items():
            if votes >= quota and candidate not in elected:
                elected.append(candidate)
                # Distribute surplus votes
                surplus = votes - quota
                if surplus > 0:
                    for rmap, count in zip(rmaps, rmapscounts):
                        if rmap.get(candidate) == 1:
                            # Distribute surplus votes proportionally
                            for pref_candidate in rmap:
                                if pref_candidate != candidate:
                                    first_prefs[pref_candidate] += (count * surplus / votes)
                                    break
                first_prefs[candidate] = 0  # Reset the votes for the elected candidate

        # If no candidate meets the quota, eliminate the candidate with the fewest votes
        if len(elected) < seats:
            min_votes = min(first_prefs.values())
            eliminate_candidates = [c for c in first_prefs if first_prefs[c] == min_votes and c not in elected]
            for candidate in eliminate_candidates:
                for rmap, count in zip(rmaps, rmapscounts):
                    if rmap.get(candidate) == 1:
                        for pref_candidate in rmap:
                            if pref_candidate != candidate:
                                first_prefs[pref_candidate] += count
                                break
                first_prefs[candidate] = float('inf')  # Mark this candidate as eliminated

    # Map elected candidate IDs back to names
    elected_names = [cmap[c] for c in elected]
    return elected_names

cmap = {1: 'Candidate 1', 2: 'Candidate 2', 3: 'Candidate 3', 4: 'Candidate 4', 5: 'Candidate 5', 6: 'Candidate 6'}
rmaps = [{2: 1, 5: 2, 1: 3, 6: 4, 3: 5, 4: 6}, {6: 1, 3: 2, 1: 3, 5: 4, 4: 5, 2: 6}, {1: 1, 5: 2, 3: 3, 4: 4, 2: 5, 6: 6}, {5: 1, 4: 2, 3: 3, 1: 4, 2: 5, 6: 6}, {5: 1, 6: 2, 4: 3, 3: 4, 2: 5, 1: 6}, {5: 1, 1: 2, 2: 3, 4: 4, 3: 5, 6: 6}, {1: 1, 3: 2, 2: 3, 4: 4, 5: 5, 6: 6}, {6: 1, 5: 2, 2: 3, 1: 4, 4: 5, 3: 6}, {1: 1, 6: 2, 2: 3, 3: 4, 5: 5, 4: 6}, {6: 1, 1: 2, 2: 3, 4: 4, 3: 5, 5: 6}, {5: 1, 1: 2, 3: 3, 2: 4, 4: 5, 6: 6}, {1: 1, 5: 2, 2: 3, 6: 4, 4: 5, 3: 6}, {2: 1, 6: 2, 3: 3, 4: 4, 5: 5, 1: 6}, {2: 1, 5: 2, 4: 3, 3: 4, 6: 5, 1: 6}, {2: 1, 1: 2, 4: 3, 5: 4, 3: 5, 6: 6}, {3: 1, 5: 2, 4: 3, 2: 4, 6: 5, 1: 6}, {6: 1, 1: 2, 5: 3, 3: 4, 4: 5, 2: 6}, {2: 1, 6: 2, 3: 3, 1: 4, 4: 5, 5: 6}, {2: 1, 4: 2, 1: 3, 3: 4, 5: 5, 6: 6}, {2: 1, 1: 2, 6: 3, 5: 4, 3: 5, 4: 6}, {5: 1, 1: 2, 3: 3, 6: 4, 4: 5, 2: 6}, {5: 1, 6: 2, 2: 3, 3: 4, 4: 5, 1: 6}, {3: 1, 1: 2, 4: 3, 5: 4, 2: 5, 6: 6}, {6: 1, 1: 2, 3: 3, 5: 4, 4: 5, 2: 6}, {3: 1, 5: 2, 2: 3, 1: 4, 6: 5, 4: 6}, {1: 1, 3: 2, 5: 3, 2: 4, 4: 5, 6: 6}, {5: 1, 6: 2, 1: 3, 4: 4, 2: 5, 3: 6}, {1: 1, 4: 2, 2: 3, 3: 4, 5: 5, 6: 6}, {2: 1, 5: 2, 4: 3, 6: 4, 3: 5, 1: 6}, {5: 1, 1: 2, 2: 3, 4: 4, 6: 5, 3: 6}, {4: 1, 5: 2, 1: 3, 3: 4, 6: 5, 2: 6}, {2: 1, 3: 2, 4: 3, 1: 4, 6: 5, 5: 6}, {6: 1, 3: 2, 1: 3, 4: 4, 2: 5, 5: 6}, {3: 1, 4: 2, 6: 3, 1: 4, 2: 5, 5: 6}, {1: 1, 4: 2, 6: 3, 5: 4, 3: 5, 2: 6}, {3: 1, 1: 2, 4: 3, 2: 4, 6: 5, 5: 6}, {4: 1, 6: 2, 5: 3, 2: 4, 3: 5, 1: 6}, {1: 1, 6: 2, 5: 3, 2: 4, 3: 5, 4: 6}, {4: 1, 3: 2, 2: 3, 5: 4, 1: 5, 6: 6}, {5: 1, 6: 2, 3: 3, 2: 4, 1: 5, 4: 6}, {4: 1, 3: 2, 1: 3, 6: 4, 5: 5, 2: 6}, {2: 1, 1: 2, 4: 3, 5: 4, 6: 5, 3: 6}, {6: 1, 3: 2, 1: 3, 2: 4, 5: 5, 4: 6}, {5: 1, 2: 2, 6: 3, 1: 4, 4: 5, 3: 6}, {6: 1, 5: 2, 2: 3, 1: 4, 3: 5, 4: 6}, {6: 1, 5: 2, 4: 3, 3: 4, 1: 5, 2: 6}, {2: 1, 3: 2, 5: 3, 4: 4, 6: 5, 1: 6}, {4: 1, 6: 2, 2: 3, 3: 4, 5: 5, 1: 6}, {1: 1, 3: 2, 6: 3, 4: 4, 5: 5, 2: 6}, {6: 1, 1: 2, 4: 3, 2: 4, 3: 5, 5: 6}, {4: 1, 1: 2, 3: 3, 2: 4, 6: 5, 5: 6}, {3: 1, 2: 2, 1: 3, 6: 4, 5: 5, 4: 6}, {4: 1, 5: 2, 6: 3, 2: 4, 1: 5, 3: 6}, {4: 1, 1: 2, 5: 3, 2: 4, 6: 5, 3: 6}, {2: 1, 4: 2, 1: 3, 5: 4, 6: 5, 3: 6}, {6: 1, 4: 2, 5: 3, 2: 4, 3: 5, 1: 6}, {6: 1, 1: 2, 5: 3, 3: 4, 2: 5, 4: 6}, {6: 1, 4: 2, 3: 3, 5: 4, 1: 5, 2: 6}, {6: 1, 4: 2, 3: 3, 2: 4, 1: 5, 5: 6}, {4: 1, 2: 2, 1: 3, 5: 4, 6: 5, 3: 6}, {3: 1, 1: 2, 5: 3, 4: 4, 2: 5, 6: 6}, {3: 1, 4: 2, 2: 3, 1: 4, 6: 5, 5: 6}, {5: 1, 4: 2, 6: 3, 1: 4, 3: 5, 2: 6}, {3: 1, 4: 2, 2: 3, 6: 4, 1: 5, 5: 6}, {1: 1, 2: 2, 4: 3, 3: 4, 6: 5, 5: 6}, {1: 1, 4: 2, 3: 3, 2: 4, 5: 5, 6: 6}, {2: 1, 4: 2, 5: 3, 1: 4, 6: 5, 3: 6}, {4: 1, 3: 2, 1: 3, 2: 4, 6: 5, 5: 6}, {1: 1, 6: 2, 4: 3, 2: 4, 5: 5, 3: 6}, {5: 1, 2: 2, 3: 3, 1: 4, 6: 5, 4: 6}, {3: 1, 1: 2, 2: 3, 5: 4, 4: 5, 6: 6}, {2: 1, 5: 2, 1: 3, 3: 4, 4: 5, 6: 6}, {2: 1, 6: 2, 4: 3, 1: 4, 3: 5, 5: 6}, {2: 1, 4: 2, 6: 3, 3: 4, 5: 5, 1: 6}, {5: 1, 2: 2, 4: 3, 6: 4, 3: 5, 1: 6}, {2: 1, 1: 2, 5: 3, 6: 4, 3: 5, 4: 6}, {3: 1, 5: 2, 2: 3, 6: 4, 1: 5, 4: 6}, {3: 1, 6: 2, 1: 3, 5: 4, 2: 5, 4: 6}, {6: 1, 5: 2, 1: 3, 4: 4, 2: 5, 3: 6}, {1: 1, 3: 2, 5: 3, 2: 4, 6: 5, 4: 6}, {3: 1, 1: 2, 2: 3, 6: 4, 4: 5, 5: 6}, {1: 1, 3: 2, 2: 3, 5: 4, 4: 5, 6: 6}, {3: 1, 2: 2, 5: 3, 1: 4, 6: 5, 4: 6}, {1: 1, 2: 2, 3: 3, 5: 4, 4: 5, 6: 6}, {4: 1, 5: 2, 6: 3, 1: 4, 2: 5, 3: 6}, {6: 1, 1: 2, 5: 3, 2: 4, 3: 5, 4: 6}, {5: 1, 2: 2, 4: 3, 1: 4, 3: 5, 6: 6}, {3: 1, 6: 2, 5: 3, 2: 4, 4: 5, 1: 6}, {6: 1, 1: 2, 2: 3, 5: 4, 3: 5, 4: 6}, {5: 1, 3: 2, 6: 3, 1: 4, 2: 5, 4: 6}, {6: 1, 3: 2, 5: 3, 4: 4, 2: 5, 1: 6}, {2: 1, 4: 2, 1: 3, 3: 4, 6: 5, 5: 6}, {4: 1, 6: 2, 5: 3, 1: 4, 2: 5, 3: 6}, {4: 1, 2: 2, 5: 3, 1: 4, 6: 5, 3: 6}]
rmapscounts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

winners = stv(cmap, rmaps, rmapscounts)
print(winners)