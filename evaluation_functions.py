import numpy as np
from sklearn.metrics import roc_curve
from itertools import combinations
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


class EvaluateIdentification:

    def __init__(self, X, y, sim_fct, num_imposter=10):

        # create genuine/imposter ID_sets
        self._genuine_ID_sets = self._createGenuineIDs(y)
        self._imposter_ID_sets = self._createImposterIDs(y, num_imposter)

        # compute scores from the ID_sets
        # genuine
        self._scores_gen = self._computeScores(self._genuine_ID_sets, X, sim_fct)
        self._y_gen = np.ones(len(self._scores_gen))
        # imposter
        self._scores_imp = self._computeScores(self._imposter_ID_sets, X, sim_fct)
        self._y_imp = np.zeros(len(self._scores_imp))
        # merged
        self._scores = np.hstack([self._scores_gen, self._scores_imp])
        self._y_comp = np.hstack([self._y_gen, self._y_imp])

        # compute ROC
        self._far, self._tar, self._thresholds = roc_curve(self._y_comp, self._scores)

    ### evaluation functions

    def getROC(self):
        """ Return ROC information
        """
        return self._far, self._tar, self._thresholds

    def computeEER(self):
        """ Compute the equal error rate
        """
        values = np.abs(1 - self._tar - self._far)
        idx = np.argmin(values)
        eer = self._far[idx]
        return eer

    def computeTAR(self, FAR_threshold):
        """ Returns the true acceptance rate at a certain FAR threshold
        """
        values = np.abs(self._far - FAR_threshold)
        idx = np.argmin(values)
        return self._tar[idx]

    def computeThreshold(self, FAR_threshold):
        """ Returns the true acceptance rate at a certain FAR threshold
        """
        values = np.abs(self._thresholds - FAR_threshold)
        idx = np.argmin(values)
        return self._thresholds[idx]

    def getScoreDistribution(self):
        """ Returns the genuine and imposter scores
        """
        return self._scores_gen, self._scores_imp

    ### helper functions

    def _createGenuineIDs(self, y):
        """Create genuine ID_sets
        """
        genuine_ID_sets = []
        for i in set(y):
            idx = np.argwhere(y == i)
            genuine_ID_sets.append(idx)
        return genuine_ID_sets

    def _createImposterIDs(self, y, num_imposter=10):
        np.random.seed(1)
        """ Create imposter ID_sets
        """
        imposter_ID_sets = []
        for i in set(y):
            idx = np.argwhere(y != i)
            np.random.shuffle(idx)
            idx_reduced = idx[:num_imposter]
            imposter_ID_sets.append(idx_reduced)
        return imposter_ID_sets

    def _computeScores(self, ID_sets, X, sim_fct):
        """ Compute the scores from the similarity function sim_fct

            ID_sets: list of ID_sets from the genuines/imposter
            X: data matrix
        """
        scores = []
        for i in range(len(ID_sets)):
            combs = combinations(ID_sets[i], 2)
            for comb in combs:
                x_probe, x_ref = X[comb[0], :], X[comb[1], :]
                score = sim_fct(x_probe, x_ref)
                scores.append(score)
        return np.array(scores)


# find fmr and fnmr at specific threshold
def find_FMR_FNMR(thr_value, thr_list, fmr_list, fnmr_list):
    l = np.abs(thr_list - thr_value)
    thr_idx = np.argmin(l)
    fmr = fmr_list[thr_idx]
    fnmr = fnmr_list[thr_idx]
    return (fmr, fnmr)


# compute fdr
def compute_fdr(A_tau, B_tau, alpha=0.5):
    max_A_tau = max([abs(x - y) for x in A_tau for y in A_tau])
    max_B_tau = max([abs(x - y) for x in B_tau for y in B_tau])
    return 1 - ((alpha * max_A_tau + (1 - alpha) * max_B_tau))


# get similarity score from logistic regression
def logistic(a, b):
    X = np.abs(a - b)
    return (np.array(pd.DataFrame(model.predict_proba(X))[1]).item())


### cosine similarity
def cos_sim(a, b):
    """ Cosine similarity between vector a and b
    """
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class Evaluate:
    def __init__(self, X, y, label, sim_fct):
        # take embeddings for different ethnicities
        self.white_emb = X[label == 'White']
        self.other_emb = X[label == 'Other']
        self.afr_emb = X[label == 'Black-or-African-American']
        self.asian_emb = X[label == 'Asian']
        # evaluate identification for all embeddings and separately for ethnicity
        self.ei = EvaluateIdentification(X, y, sim_fct, num_imposter=10)
        self.ei_w = EvaluateIdentification(self.white_emb, y[label == 'White'], sim_fct, num_imposter=10)
        self.ei_o = EvaluateIdentification(self.other_emb, y[label == 'Other'], sim_fct, num_imposter=10)
        self.ei_af = EvaluateIdentification(self.afr_emb, y[label == 'Black-or-African-American'], sim_fct,
                                            num_imposter=10)
        self.ei_as = EvaluateIdentification(self.asian_emb, y[label == 'Asian'], sim_fct, num_imposter=10)
        # get fpr, tpr and threshold
        self.fpr, self.tpr, self.thr = self.ei.getROC()
        self.fpr_w, self.tpr_w, self.thr_w = self.ei_w.getROC()
        self.fpr_o, self.tpr_o, self.thr_o = self.ei_o.getROC()
        self.fpr_af, self.tpr_af, self.thr_af = self.ei_af.getROC()
        self.fpr_as, self.tpr_as, self.thr_as = self.ei_as.getROC()

    # plot ROC curve
    def plot_curve(self):
        plt.figure()
        plt.plot(self.fpr, self.tpr)
        plt.plot(self.fpr_w, self.tpr_w)
        plt.plot(self.fpr_o, self.tpr_o)
        plt.plot(self.fpr_af, self.tpr_af)
        plt.plot(self.fpr_as, self.tpr_as)
        plt.xscale('log')
        plt.xlabel('FMR')
        plt.ylabel('1-FNMR')
        plt.legend(['Overall', 'White', 'Other', 'African-American', 'Asian'])
        plt.grid()

    # get fnmr at a specific fmr
    def fnmr(self, fmr):
        """
        false non-match rate
            measures the proportion of face comparisons of the same identity, or mated face pairs, that do not result in
            a match.

        :param fmr: False Match Rate threshold at which the FNMR will be computed
        :return: Print of the overall FNMR, as well as for the different ethnicities separately
        """
        # computeTAR = compute True Acceptance Rate overall and for all ethnicities separately
        self.tpr_specific = self.ei.computeTAR(fmr)
        self.tpr_specific_w = self.ei_w.computeTAR(fmr)
        self.tpr_specific_o = self.ei_o.computeTAR(fmr)
        self.tpr_specific_af = self.ei_af.computeTAR(fmr)
        self.tpr_specific_as = self.ei_as.computeTAR(fmr)
        print('overall fnmr at fmr of', fmr, ':', 1 - self.tpr_specific, '\n',
              'white fnmr at fmr of', fmr, ':', 1 - self.tpr_specific_w, '\n',
              'other fnmr at fmr of', fmr, ':', 1 - self.tpr_specific_o, '\n',
              'asian fnmr at fmr of', fmr, ':', 1 - self.tpr_specific_as, '\n',
              'africa-ameriacan fnmr at fmr of', fmr, ':', 1 - self.tpr_specific_af)

    # get eer
    def eer(self):
        eer = self.ei.computeEER()
        eer_w = self.ei_w.computeEER()
        eer_o = self.ei_o.computeEER()
        eer_af = self.ei_af.computeEER()
        eer_as = self.ei_as.computeEER()
        print('overall :', eer)
        print('white :', eer_w)
        print('other :', eer_o)
        print('african-american :', eer_af)
        print('asian :', eer_as)

    # get FDR
    def get_FDR(self, fmr):
        t = self.ei.computeThreshold(fmr)
        fmr_o, fnmr_o = find_FMR_FNMR(t, self.fpr_o, 1 - self.tpr_o, self.thr_o)
        fmr_w, fnmr_w = find_FMR_FNMR(t, self.fpr_w, 1 - self.tpr_w, self.thr_w)
        fmr_af, fnmr_af = find_FMR_FNMR(t, self.fpr_af, 1 - self.tpr_af, self.thr_af)
        fmr_as, fnmr_as = find_FMR_FNMR(t, self.fpr_as, 1 - self.tpr_as, self.thr_as)
        A_tau = [fmr_o, fmr_w, fmr_af, fmr_as]
        B_tau = [fnmr_o, fnmr_w, fnmr_af, fnmr_as]
        return (compute_fdr(A_tau, B_tau))

    # get overall measure
    def overall_score(self):
        return (0.5 * self.get_FDR(0.1) / 0.9689 + (1 - 0.5) * (self.ei.computeTAR(0.1)) / (1 - 0.0205))