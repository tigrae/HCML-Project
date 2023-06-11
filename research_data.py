from evaluation_functions import *
import matplotlib.pyplot as plt
import math
import sys

test_embedding = np.load('data/test_emb.npy')
test_ethnicity = np.load('data/test_ethnicity.npy')
test_identities = np.load('data/test_id.npy')


def count_identities_per_ethnicity(test_ethnicity, test_identities):
    ethnicity_identities = {
        "White": set(),
        "Black-or-African-American": set(),
        "Asian": set(),
        "Other": set()
    }

    for ethnicity, identity in zip(test_ethnicity, test_identities):
        ethnicity_identities[ethnicity].add(identity)

    num_identities = {
        ethnicity: len(identities)
        for ethnicity, identities in ethnicity_identities.items()
    }

    print(num_identities)


def cos_sim(a, b):
    """ Cosine similarity between vector a and b
    """
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def print_class_attributes(given_class):
    # Get a list of all attributes
    attributes = dir(given_class)

    # Filter out the attributes that start with underscore (usually private attributes)
    set_attributes = [attr for attr in attributes if not attr.startswith('_')]

    # Print the names of all set attributes
    for attr in set_attributes:
        print(attr)


def euclidian_distance(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return math.sqrt((sum(pow(a2-b2, 2) for a2, b2 in zip(a, b))))


def euclidean_dist_simpler(a, b):
    """ Euclidean distance between vector a and b
    """
    return np.linalg.norm(a - b)


if __name__ == "__main__":

    """
    Approach:
        - First we get the genuine ID sets, which are sets of the embedding indices that belong to the same identity 
        - After that we create combinations of indices belonging to the same face
        - for those, we compute the difference between each value of the embedding
        - we calculate the average of all difference arrays we get to find out which values are more close for 
          embeddings of the same identity
        - we use the resulting vector as weight vector over all features in the similarity function
    """

    # get genuine ID sets from Evaluate class
    eval = Evaluate(test_embedding, test_identities, test_ethnicity, cos_sim)
    genuine_ID_sets = eval.ei._genuine_ID_sets


    for i in range(len(genuine_ID_sets)):
        combs = combinations(genuine_ID_sets[i], 2)
        for comb in combs:
            print(comb)
            # todo: calculate difference between each, sum up, take average after

    pass


    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    #
    # print(euclidian_distance(a, b))
    # print(euclidean_dist_simpler(a, b))
