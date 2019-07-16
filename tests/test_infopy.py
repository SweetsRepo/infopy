from infopy.metrics import *

# Special imports for testing private functions
from infopy.metrics import _balance_distributions
from infopy.metrics import _to_tuple

def test_distribution_balancing():
    """
    Only needs to be performed once for intersection, kld, and jsd
    """
    # Test inputs with different domains
    a = [1,1,1]
    b = [1,2,3]
    a_bal, b_bal = _balance_distributions(a,b)
    a_key_set = set(a_bal.keys())
    b_key_set = set(b_bal.keys())
    assert a_key_set == b_key_set

def test_intersection():
    """
    Intersection of Histograms Testing
    """
    # Test raw inputs
    a = [1,1,2]
    b = [1,2,2]
    assert intersection(a,b) == 2/3

    # Test probability inputs
    a = {1:2, 2:1}
    b = {1:1, 2:2}
    assert intersection(a,b) == 2/3

def test_kld():
    """
    Kullback Leibler Divergence Testing
    """
    # Test raw inputs
    a = [1,1,2]
    b = [1,2,2]

    # Analytically computed result
    analytical = 0.23105
    assert analytical-analytical*.1 <= kl_divergence(a,b) <= analytical+analytical*.1

    # Test probability inputs
    a = {1:2, 2:1}
    b = {1:1, 2:2}
    assert analytical-analytical*.1 <= kl_divergence(a,b) <= analytical+analytical*.1

def test_jsd():
    """
    Jensen Shannon Divergence Testing
    """
    # Test raw inputs
    a = [1,1,2]
    b = [1,2,2]

    # Analytically computed result
    analytical = 0.05663
    assert analytical-analytical*.1 <= js_divergence(a,b) <= analytical+analytical*.1

    # Test probability inputs
    a = {1:2, 2:1}
    b = {1:1, 2:2}
    assert analytical-analytical*.1 <= js_divergence(a,b) <= analytical+analytical*.1

def test_entropy():
    """
    Entropy Computation and Normalization Testing
    """
    a = [1,2,3,4]

    # Analytically computed result
    analytical = 1.38629
    assert analytical-analytical*.1 <= entropy(a) <= analytical+analytical*.1

    # Normalizes Distributions with Finite Support by using Uniform Distribution
    assert entropy(a, normalize=True) == 1

def test_joint():
    """
    Joint Probability Distribution Testing

    Ensures that probability may be scaled to n-many feature distributions
    """
    a = [1,2,3,4]

    # Hash should be identical given the two arrays are comprised of the same elements and are in the same order
    assert hash(_to_tuple(joint_probability(a))) == hash(_to_tuple(np.asarray([1/4 for _ in range(4)])))

    # Test multi-feature samples
    a = [[1,1], [1,2], [2,1], [2,2], [2,2]]
    assert hash(_to_tuple(joint_probability(a))) == hash(_to_tuple(np.asarray([0.2, 0.2, 0.2, 0.4])))
# 
# def test_conditional():
#     """
#     Tests Conditional Probability Computation with Input Condition Weight Term
#     """
#     a = [[1,1], [1,2], [2,1], [2,2], [2,2]]
#
#     print(conditional_probability(a))
#
# def test_weighted_conditional_entropy():
#     assert False
