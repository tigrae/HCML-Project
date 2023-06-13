from evaluation_functions import *
import matplotlib.pyplot as plt
import math
import sys

dev_embedding = np.load('data/development_emb.npy')
dev_ethnicity = np.load('data/development_ethnicity.npy')
dev_identities = np.load('data/development_id.npy')


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
    eval = Evaluate(dev_embedding, dev_identities, dev_ethnicity, cos_sim)
    genuine_ID_sets = eval.ei._genuine_ID_sets

    # iterate over each genuine id_Set
    total_diff = np.zeros((1, 512))
    for i in range(len(genuine_ID_sets)):

        # get all possible combinations of the current genuine id set
        combs = [(genuine_ID_sets[i][j], genuine_ID_sets[i][k])
                 for j in range(len(genuine_ID_sets[i]))
                 for k in range(j + 1, len(genuine_ID_sets[i]))]

        # iterate over all combinations and sum of the differences between all features of the two embeddings
        set_diff = np.zeros((1, 512))
        for comb in combs:

            # get the two embeddings belonging to the
            emb1, emb2 = test_embedding[comb[0]], test_embedding[comb[1]]

            # assert that identities truly are the same
            assert test_identities[comb[0]] == test_identities[comb[1]], \
                f"Assertion failed: Identities at indices {comb[0]} and {comb[1]} are not equal."

            # add to combinu
            set_diff += np.abs(emb1 - emb2)

            pass

        # average set differences and add them to total differences
        set_diff /= len(combs)
        total_diff += set_diff

    # average the total differences over all sets
    total_diff /= len(genuine_ID_sets)
    pass



    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 6])
    #
    # print(euclidian_distance(a, b))
    # print(euclidean_dist_simpler(a, b))
