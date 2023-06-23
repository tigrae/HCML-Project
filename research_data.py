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


def calculate_difference_array(id_sets, genuine):

    # iterate over each genuine id_Set
    total_diff = np.zeros((1, 512))
    for i in range(len(id_sets)):

        # get all possible combinations of the current genuine id set
        combs = [(id_sets[i][j], id_sets[i][k])
                 for j in range(len(id_sets[i]))
                 for k in range(j + 1, len(id_sets[i]))]

        # iterate over all combinations and sum of the differences between all features of the two embeddings
        set_diff = np.zeros((1, 512))
        for comb in combs:

            # get the two embeddings belonging to the
            emb1, emb2 = dev_embedding[comb[0]], dev_embedding[comb[1]]

            # # assert that identities are genuine or imposters
            # if genuine:
            #     assert dev_identities[comb[0]] == dev_identities[comb[1]], \
            #         f"Assertion failed: Identities at indices {comb[0]} and {comb[1]} are not equal for genuine set."
            # else:
            #     print(f"{dev_identities[comb[0]]} and {dev_identities[comb[1]]}")
            #     assert dev_identities[comb[0]] != dev_identities[comb[1]], \
            #         f"Assertion failed: Identities at indices {comb[0]} and {comb[1]} are equal for imposter set."

            # add to combinu
            set_diff += np.abs(emb1 - emb2)

        # average set differences and add them to total differences
        set_diff /= len(combs)
        total_diff += set_diff

    # average the total differences over all sets
    total_diff /= len(id_sets)

    return total_diff


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
    eval_dev = Evaluate(dev_embedding, dev_identities, dev_ethnicity, cos_sim)
    genuine_id_sets = eval_dev.ei._genuine_ID_sets
    imposter_id_sets = eval_dev.ei._imposter_ID_sets

    genuine_wights = calculate_difference_array(genuine_id_sets, True)
    imposter_weights = calculate_difference_array(imposter_id_sets, False)

    print(len(genuine_wights))
    print(len(imposter_weights))

    pass

    genuine_wights = 1-((genuine_wights - 0.02324784459798256) * 1000 / 5.355507872984975)

    np.save("genuine_weights.npy", genuine_wights)

    pass
