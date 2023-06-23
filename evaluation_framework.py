from evaluation_functions import *
import matplotlib.pyplot as plt
import math
import numpy as np
import sys

test_embedding = np.load('data/test_emb.npy')
test_ethnicity = np.load('data/test_ethnicity.npy')
test_identities = np.load('data/test_id.npy')


def sigmoid(x):
    return ((1/(1 + np.exp(-x)))-0.0)*2

def cos_sim(a, b):
    """ Cosine similarity between vector a and b
    """
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidian_distance(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return math.sqrt((sum(pow(a2-b2, 2) for a2, b2 in zip(a, b))))


def euclidean_dist_simpler(a, b):
    """ Euclidean distance between vector a and b
    """
    return np.linalg.norm(a - b)


def probant(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return random.random()


def myFunc2(a,b):
    a, b = a.reshape(-1), b.reshape(-1)
    sumOfDifs = 0
    for i in range(0,len(a)):
        dif = pow(a[i]-b[i],2)/2
        sumOfDifs = sumOfDifs + dif
    return max(1-(pow(float(sumOfDifs),0.5)/len(a)),0)


def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([sigmoid(((a - b) ** 2) / (a + b)) for (a, b) in zip(A, B)])
    return 1 - math.log10(abs(chi))


def quartel(a,b):
    a, b = a.reshape(-1), b.reshape(-1)
    sumOfDifs = 0
    for i in range(0,len(a)):
        dif = pow(a[i]-b[i], 2)
        sumOfDifs = sumOfDifs + dif
    a.sort()
    b.sort()
    AuQuartel = a[len(a)//4]
    AoQuartel = a[3*len(a)//4]
    BuQuartel = b[len(b)//4]
    BoQuartel = b[3*len(b)//4]
    return 1-(pow(sumOfDifs, 0.5)/len(a))-0.002*(abs(np.median(a)-np.median(b))+abs(AoQuartel-BoQuartel)+abs(AuQuartel-BuQuartel))


def squared_dist(a,b):
    a, b = a.reshape(-1), b.reshape(-1)
    sumOfDifs = 0
    for i in range(0,len(a)):
        dif = pow(a[i]-b[i],2)
        sumOfDifs = sumOfDifs + dif
    return (1-(pow(sumOfDifs,0.5)/len(a)))


"""
THIS IS THE FUNCTION. WOWOW!
"""
def similarity_with_weights(a, b):
    weights_array = np.load("gen_imp_weights_50_50.npy")
    a *= weights_array
    b *= weights_array
    return squared_dist(a, b)
# with genuine_weights.npy:         1.0160617901720623
# with genuine_weights_0.1.npy:     1.0160033973786078
# with genuine_weights_sinus.npy:   1.0157728199444982

# "gen_imp_weights_max.npy"         1.016513827673255
# "gen_imp_weights_80_20.npy"       1.0161534785999218
#

eval = Evaluate(test_embedding, test_identities, test_ethnicity, cos_sim)

# plot and show curve
eval.plot_curve()
plt.show()

""" equal error rate
    the location on a ROC or DET curve where the false acceptance rate and false rejection rate are equal. In general, 
    the lower the equal error rate value, the higher the accuracy of the biometric system."""
eval.eer()

""" false non-match rate
    measures the proportion of face comparisons of the same identity, or mated face pairs, that do
    not result in a match."""
print(f"fnmr: {eval.fnmr(0.1)}")

print(f"FDR: {eval.get_FDR(0.1)}")

print(eval.overall_score())
