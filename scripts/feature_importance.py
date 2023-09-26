import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import MT19937, RandomState, SeedSequence
from scipy.stats import norm
from sklearn.svm import SVC
from sklearn.utils import resample
from statsmodels.sandbox.stats.multicomp import multipletests as MCP

from .classification_functions import SiteWise_RandomUnderSampler


def compute_wsign(clf, X, y, groups, n_samples, seed):
    # Create labels to stratify (i.e. site * class combinations)
    labels_to_strat = np.array([site + "_and_" + str(int(label)) for site, label in zip(groups, y)])

    rs = RandomState(MT19937(SeedSequence(seed)))

    # Resample n_samples (Total Samples * gamma) stratified for class and site to obtain % of sample
    X_res, y_res, groups_res = resample(X, y, groups,
                                        n_samples=n_samples, replace=False, stratify=labels_to_strat,
                                        random_state=rs)

    # Now undersample minority class per site
    rus_idx = SiteWise_RandomUnderSampler(groups_res, y_res, random_state=rs)
    X_res_us = X_res[rus_idx]
    y_res_us = y_res[rus_idx]
    groups_res_us = groups_res[rus_idx]

    # Train the classifier
    clf.fit(X_res_us, y_res_us)
    w = clf.coef_.ravel()

    # Create a vector counting the number of positive weights (indirectly we are also counting the negatives)
    w_sign = np.zeros(w.shape)
    w_sign[w > 0] = 1

    return w_sign


def computeZTest(hatp, N, p0=0.5, gamma=0.5):
    s = ((1 - gamma) / gamma) * hatp * (1 - hatp)

    # To avoid divide by zero
    s[np.where(s == 0)] = np.finfo(float).eps

    z = np.divide(hatp - p0, np.sqrt(s))
    p = 2 * (1 - norm.cdf(np.abs(z)))  # WBB: Two-tailed

    # WBB: added this here so lowest p values are set to minimum of system
    p[np.where(p == 0)] = np.finfo(float).eps

    return z, p


def calc_sign_weighted_importance(X, y, groups, svm_C, FS_cols,
                                  num_resampling_iter=10000, sample_perc=0.5, n_jobs=20):
    # Array to store summed weight signs
    w_sign_tot = np.zeros((X.shape[-1],))

    # Total number of samples
    N_total_samples = X.shape[0]

    # Number of samples to resample (default is 50% of total sample)
    N_resample = N_total_samples * sample_perc

    # Define the classifier to be used for the voxel selection
    clf = SVC(C=svm_C, class_weight='balanced', kernel='linear')

    # Derive sign weights with parallelization
    w_sign = Parallel(n_jobs=n_jobs, verbose=1)(delayed(compute_wsign)
                                                (clf, X, y, groups, N_resample, i)
                                                for i in range(num_resampling_iter))

    # Sum weight signs across resampling iterations
    w_sign = np.array(w_sign)
    w_sign_tot = w_sign.sum(axis=0)

    # Estimate the probability of being positive
    hatp = w_sign_tot / num_resampling_iter

    # Transform to Z values and two-tailed p-values
    z, p_two_tailed = computeZTest(hatp, num_resampling_iter)

    # Apply MCP with FDR
    reject_H0, p_FDR_corrected, _, _ = MCP(p_two_tailed, 0.05, method='fdr_tsbh')

    # Create DataFrame of results
    sign_weighted_importance_df = pd.DataFrame(data=np.c_[FS_cols, hatp, z, p_two_tailed, p_FDR_corrected, reject_H0],
                                               columns=['Feature', 'hatp', 'Z', 'p_uncorrected', 'p_FDR_corrected',
                                                        'reject_H0_FDR'])

    sign_weighted_importance_df['abs_Z'] = np.abs(sign_weighted_importance_df.Z)

    return sign_weighted_importance_df