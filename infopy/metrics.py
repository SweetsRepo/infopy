"""
Contains several metrics for comparing feature histograms. All methods are
scalable to n-many combinations of features. Metrics include the following:

* Intersection of Histograms
* Kullback-Leibler Divergence
* Jensen-Shannon Divergence

Note: All functions using the collections.Counter interface, allowing for raw
samples or probability distributions to be used.

Additionally, methods are provided to analyze histograms individually. These
include the following:

* Entropy
* Joint Probability
* Conditional Probability

The Joint and Conditional Probability functions can be used for raw output or
fed into the entropy computing function to find their corresponding entropy
values. More details are included in each function doc-string.

"""

from collections import Counter
from itertools import combinations, permutations

import numpy as np
from scipy.stats import entropy as ent

def intersection(p_hist, q_hist):
    """
    Computes the Intersection of Histograms between P and Q.

    Arguments:
        p_hist: Iterable Feature Histogram
        q_hist: Iterable Feature Histogram

    Returns:
        inter: Intersection of score computed as shown in notes

    Notes:
        G(P,Q) = {sum_i^N(min(P_i, Q_i)) over max(sum_i^N(P), sum_i^N(Q))}

        * Works even if values from one distribution occur with 0 probability in
        the other

        * Intersection is symmetric

    """
    p_cnt, q_cnt = _balance_distributions(p_hist, q_hist)

    inter = 0
    for k,v in p_cnt.items():
        inter += min(v, q_cnt[k])
    inter /= max(sum(p_cnt.values()), sum(q_cnt.values()))

    return inter


def kl_divergence(p_hist:dict, q_hist:dict, normalize:bool=False):
    """
    Computes the Kullback Leibler Divergence betweeen P and Q.

    Arguments:
        p_hist: Iterable Feature Histogram
        q_hist: Iterable Feature Histogram
        normalize: Normalizes the result (r) by {1 over (r + 1)}

    Returns:
        kl: Kullback Leibler Divergence computed as shown in notes

    Notes:
        D_{KL}(P,Q) = sum_i^N(P_i * log({P_i over Q_i}))

        * Note that the normalize argument 'flips' the result (smaller values =>
        larger divergences).

        * KL Divergence is asymmetric

    """
    p_cnt, q_cnt = _balance_distributions(p_hist, q_hist)

    # Convert counts into probabilities
    p_dist = np.zeros(len(p_cnt))
    q_dist = np.zeros(len(q_cnt))
    for i,k in enumerate(p_cnt):
        p_dist[i] = p_cnt[k]
        q_dist[i] = q_cnt[k]

    p_dist /= sum(p_dist)
    q_dist /= sum(q_dist)

    kl = sum(p_dist * np.log((p_dist / q_dist)))
    if normalize:
        kl = 1 / (kl + 1)

    return kl


def js_divergence(p_hist:dict, q_hist:dict, normalize:bool=False):
    """
    Computes the Jensen Shannon Divergence betweeen P and Q.

    Arguments:
        p_hist: Iterable Feature Histogram
        q_hist: Iterable Feature Histogram
        normalize: Normalizes the result (r) by {1 over (r + 1)}

    Returns:
        js: Jensen Shannon Divergence computed as shown in notes

    Notes:
        M = 0.5(P + Q)
        D_{JS}(P,Q) = 0.5*(D_{KL}(P,M) + D_{KL}(Q,M))

        * Note that the normalize argument 'flips' the result (smaller values =>
        larger divergences).

        * JS Divergence is symmetric

    """
    p_cnt, q_cnt = _balance_distributions(p_hist, q_hist)

    # Create midpoint distribution (M)
    m_cnt = {}
    for k,v in p_cnt.items():
        m_cnt[k] = 0.5 * (v + q_cnt[k])

    js = 0.5 * (kl_divergence(p_cnt, m_cnt) + kl_divergence(q_cnt, m_cnt))

    if normalize:
        js = 1 / (js + 1)

    return js


def entropy(p_hist:dict, normalize:bool=False):
    """
    Computes the entropy of the provided histogram.

    Arguments:
        p_hist: Iterable Feature Histogram
        normalize: Normalizes the result by dividing by a Uniform Distribution

    Returns:
        entropy: Entropy of the Histogram

    Notes:
        * Normalization is only valid for data which has a finite support
        (Bounded Discrete Distribution).

        * Cardinality of the input histogram is used for the normalizing uniform
        distribution.

    """
    p_dist = joint_probability(p_hist)

    entropy = -1*sum(p_dist * np.log(p_dist))
    if normalize:
        size = len(p_dist)
        # Multiply a single value's entropy by the cardinality
        entropy /= (size * -1 * ((1 / size) * np.log(1/size)))
    return entropy

def joint_probability(p_hist:dict):
    """
    Computes joint probability of the features in the histograms dictionary

    Arguments:
        p_hist: Iterable Feature Histogram

    Returns:
        p_dist: Joint probability distributions

    Notes:
        If normalize == False then output raw counts of values occuring

    """
    p_cnt = Counter(_to_tuple(p_hist))
    p_dist = np.fromiter(p_cnt.values(), np.float)
    p_dist /= np.sum(p_dist)
    return p_dist
 
# def conditional_probability(histogram:dict, normalize:bool=False):
#     """
#     Computes the conditional entropy for all unique permutations given
#
#     Arguments:
#         histograms: Dictionary mapping features to probability distriubtions
#         normalize: Normalizes the results to probabilty values
#
#     Returns:
#         conditonal: Conditional Probability Distribution
#         weights: Corresponding dictionary with # of input conditioner occurrence
#
#     Notes:
#         If normalize == False then output raw counts of values occuring
#
#     """
#     conditional = {}
#     weights = {}
#
#     # Find all the unique input combinations
#     for comb in histogram:
#         conditioners = _to_tuple(sorted(comb[:,-1]))
#
#         # Create labels for target feature and conditioning feature
#         c_label = []
#         for c in conditioners:
#             label.append(c)
#         c_label = ','.join(label)
#         t_label = comb[-1]
#
#         # Only use a subset of the permutations given
#         # Order of the conditioning features doesn't matter
#         if c_label not in conditional:
#             conditional[c_label] = {}
#             weights[c_label] = {}
#         if target not in conditonal[c_label]:
#             conditional[c_label][t_label] = {}
#             weights[c_label][t_label] = {}
#             for value in histograms[comb]:
#                 conditioning_value = to_tuple(value[:-1])
#                 target_value = value[-1]
#
#                 # Construct the actual histogram
#                 if conditioning_value not in conditonal[c_label][t_label]:
#                     conditional[c_label][t_label][conditioning_value] = {}
#                     weights[c_label][t_label][conditioning_value] = 0
#                 if target_value not in conditional[c_label][t_label][conditioning_value]:
#                     conditional[c_label][t_label][conditioning_value][target_value] = 0
#                 conditional[c_label][t_label][conditioning_value][target_value] += 1
#
#     # Iterate over all the histograms and convert to distributions
#     if normalize:
#         for conditioner in conditional:
#             for target in conditional[conditioner]:
#                 for conditioning_value in conditional[conditioner][target]:
#                     total = sum(conditional[conditioner][target][conditioning_value].values())
#                     for target_value in conditional[conditioner][target][conditioning_value]:
#                         conditional[conditioner][target][conditioning_value][target_value] /= total
#
#     return conditional, weights
#
# def weighted_average_conditional_entropy(conditional:dict, weights:dict,
#     normalize:bool=False):
#     """
#     Computes the entropy for conditional probability distributions
#
#     Arguments:
#         conditional: Dictionary containing conditional probability distriubtions
#         weights: Dictionary containing number of times input conditioners occur
#         normalize: Normalizes the results to probabilty values
#
#     Returns:
#         wce: Weighted Conditional Entropy
#
#     Notes:
#         H_{Y|X_0,...,X_m} = sum_i^{N} {{|w_i|} over {|w|}} *
#             sum_j=^{Z}{p_{i|j} * log {1 over{p_{i|j}}}
#
#         * Weights the entropy calculation by the probability that each input
#         conditioning value(s) occurs
#
#     """
#     wce = {}
#     for conditioner in conditonal:
#         for target in conditional[conditioner]:
#             label = "%s|%s" % (target, conditioner)
#             distribution = conditional[conditioner][target]
#             entropies = []
#             unique_outputs = set()
#             for inputs in distribution:
#                 probabilities = np.fromiter(distribution[inputs].values(), float)
#                 weighting = weights[conditioner][target][inputs] / sum(weights[conditioner][target][inputs].values())
#                 entropies.append(weighting * (-1 * sum(probabilities * np.log(probabilities))))
#                 for output in distribution[inputs]:
#                     unique_outputs.add(output)
#             wce[label] = sum(entropies)
#
#             if normalize:
#                 num_unique_outputs = len(unique_outputs)
#                 # Attempt to normalize the values. Deterministic cases throw ValueError
#                 try:
#                     wce[label] /= (num_unique_outputs * -1 * ((1/num_unique_outputs) * np.log(1/num_unique_outputs)))
#                 except ValueError:
#                     wce[label] = 0
#     return wce

def _balance_distributions(p, q):
    """
    Takes in the raw histograms, converts them into counts, and balances them
    out if values in one dictionary don't occur in the other

    Arguments:
        p: Iterable collection of values
        q: Iterable collection of values

    Returns:
        p_cnt_balanced: Balanced histogram
        q_cnt_balanced: Balanced histogram

    """
    p_cnt = Counter(p)
    q_cnt = Counter(q)

    # Balance out counts to matching sets of features
    p_cnt_balanced = {}
    q_cnt_balanced = {}

    for item in q_cnt:
        if item in p_cnt:
            p_cnt_balanced[item] = p_cnt[item]
        else:
            p_cnt_balanced[item] = 0

    for item in p_cnt_balanced:
        if item in q_cnt:
            q_cnt_balanced[item] = q_cnt[item]
        else:
            q_cnt_balanced[item] = 0

    return p_cnt_balanced, q_cnt_balanced

def _get_feature_combinations(features:list):
    """
    Get's all feature overlap regions for feature pairs

    Arguments:
        features: Data as numpy array where each row is a list of alert features

    Returns:
        combination_hierarchy: Dictionary of all unique feature combinations
        with varying overlap

    Notes:
        * (e.g.) Given an Alert a = {0,1,2,...,5} and overlap size of 2
        combination_indices = [(1,2), (1,3), (1,4), ...] for all unique
        combinations and the feature hierarchy would store it as
        fh[overlap_size][comb_index] = feature_combos
    """
    overlap_size = len(features[0])
    num_features = [i for i in range(overlap_size)]
    combination_hierarchy = {i+1:{} for i in range(overlap_size)}
    while overlap_size > 0:
        feature_combos = list(combinations(num_features, overlap_size))
        for comb in feature_combos:
            combination_hierarchy[overlap_size][comb] = features[:,comb]
        overlap_size -= 1
    return combination_hierarchy

def _get_feature_permutations(features:list):
    """
    Get's all feature overlap regions for feature pairs

    Arguments:
        features: Data as numpy array where each row is a list of alert features

    Returns:
        combination_hierarchy: Dictionary of all unique feature permutations
        with varying overlap

    :notes:
        * (e.g.) Given an Alert a = {0,1,2,...,5} and overlap size of 2
        combination_indices = [(1,2), (1,3), (1,4), ...] for all unique
        permutations and the feature hierarchy would store it as
        fh[overlap_size][comb_index] = feature_combos
    """
    overlap_size = len(features[0])
    num_features = [i for i in range(overlap_size)]
    combination_hierarchy = {i+1:{} for i in range(overlap_size)}
    while overlap_size > 0:
        feature_combos = list(permutations(num_features, overlap_size))
        for comb in feature_combos:
            combination_hierarchy[overlap_size][comb] = features[:,comb]
        overlap_size -= 1
    return combination_hierarchy

def _to_tuple(iter):
    """
    Converts nested list, arrays, etc into tuples

    Arguments:
        iter: Iterable to convert into tuple

    """
    try:
        return tuple(_to_tuple(i) for i in iter)
    except TypeError:
        return iter
