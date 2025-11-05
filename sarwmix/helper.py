# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

def compare_most_best_vectors(vecA, vecB):
    """
    Compare two 12-element tie-break vectors in a descending order sense.
    Returns True if vecA is better (larger) than vecB lexicographically.
    """
    # Just do a lexicographic check:
    # for xA, xB in zip(vecA, vecB):
    #   if xA > xB: return True
    #   elif xA < xB: return False
    # if all equal, return False
    for xA, xB in zip(vecA, vecB):
        if xA > xB:
            return True
        elif xA < xB:
            return False
    return False  # they are exactly equal

def build_pairwise_tie_break_vectors(new_metrics, old_metrics):
    """
    new_metrics & old_metrics: dicts with keys:
      [ap_macro, ap_micro, f1_macro, f1_micro, prec_macro, prec_micro]

    Return (vector_new, vector_old) each a 12-element tuple:
      (#rank=1 total, #rank=1 macro,
       #rank=2 total, #rank=2 macro,
       #rank=3 total, #rank=3 macro,
       #rank=4 total, #rank=4 macro,
       #rank=5 total, #rank=5 macro,
       #rank=6 total, #rank=6 macro)
    """
    metric_order = [
        ('ap_macro', True),
        ('ap_micro', False),
        ('f1_macro', True),
        ('f1_micro', False),
        ('prec_macro', True),
        ('prec_micro', False),
    ]
    # above I've indicated which are 'macro' metrics with a boolean

    # ranks for new / old: dict from metric -> rank=1..6
    # but realistically with only 2 items, you only see rank=1 or rank=2 or ties => both rank=1
    ranks_new = {}
    ranks_old = {}

    for (m, is_macro) in metric_order:
        val_n = new_metrics[m]
        val_o = old_metrics[m]
        if val_n > val_o:
            ranks_new[m] = 1
            ranks_old[m] = 2
        elif val_n < val_o:
            ranks_new[m] = 2
            ranks_old[m] = 1
        else:  # tie
            ranks_new[m] = 1
            ranks_old[m] = 1

    def build_vector(ranks_dict):
        # counts for each rank in 1..6 => (total, macro)
        # e.g. how many metrics have rank=1, how many of those are macro, how many rank=2, etc.
        out = []
        for r in range(1, 7):  # rank 1..6
            count_total = 0
            count_macro = 0
            for (m, is_macro) in metric_order:
                if ranks_dict[m] == r:
                    count_total += 1
                    if is_macro:
                        count_macro += 1
            out.append(count_total)
            out.append(count_macro)
        return tuple(out)

    vec_new = build_vector(ranks_new)
    vec_old = build_vector(ranks_old)
    return (vec_new, vec_old)

def is_new_better_mbr(new_metrics, old_metrics):
    """
    Compare two sets of 6 metrics using the "most best results" tie-break logic, pairwise.
    Returns True if new_metrics is strictly better, else False.
    """
    vec_new, vec_old = build_pairwise_tie_break_vectors(new_metrics, old_metrics)
    return compare_most_best_vectors(vec_new, vec_old)

