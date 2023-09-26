
import os
import warnings
from time import time

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.sandbox.stats.multicomp import multipletests as MCP
from tqdm.notebook import tqdm

from .evaluation_classifier import Evaluater
from .helper_functions import has_N_per_class


def leave_one_site_out_splits(y, groups):
    cv_splits = []
    logo_CV = LeaveOneGroupOut()
    for i, (train, test) in enumerate(logo_CV.split(y=y, X=np.zeros_like(y), groups=groups)):
        cv_splits.append((train, test))
    return cv_splits


def repeated_group_stratified_KFold_splits(y, groups, n_folds=5, n_repeats=5, seed=0, verbose=True):
    # Note: if there are (y * groups) combinations with N < n_folds, these will be omitted randomly in either
    # train or test folds.

    rs = RandomState(MT19937(SeedSequence(seed)))

    # Combine class and site labels
    labels_to_strat = np.array([site + "_and_" + str(int(label)) for site, label in zip(groups, y)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create and return repeated group stratified KFold splits
        cv_splits = []
        ssKFold_cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=rs)
        for i, (train, test) in enumerate(ssKFold_cv.split(y=labels_to_strat, X=np.zeros_like(labels_to_strat))):
            cv_splits.append((train, test))
        return cv_splits


def SiteWise_RandomUnderSampler(groups, y, random_state=0):
    rus_pooled_idx = np.zeros_like(groups, dtype=bool)
    pooled_idx = np.arange(len(groups))

    for group in np.unique(groups):

        group_mask = groups == group
        y_group = y[group_mask]

        # Perform undersampling if there are two classes available for given site, otherwise drop site
        if np.unique(y_group).size > 1:
            idx_group = pooled_idx[group_mask]
            rus = RandomUnderSampler(replacement=False, random_state=random_state)
            _, _ = rus.fit_resample(y_group.reshape(-1, 1), y_group)

            rus_idx = np.sort(rus.sample_indices_)
            rus_pooled_idx[idx_group[rus_idx]] = True

    return rus_pooled_idx


def specificity(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_true == 0, dtype=float)


def site_wise_scaling(X_train, y_train, groups_train, X_test, y_test, groups_test, groups, CV='kfold'):
    with np.errstate(divide='ignore', invalid='ignore'):

        if CV == 'kfold':

            for i_g, group in enumerate(np.unique(groups)):

                # Create masks for patients and controls for training and test data for given group
                group_HC_train_mask = (groups_train == group) & (y_train == 0)
                group_train_mask = (groups_train == group)
                group_test_mask = (groups_test == group)

                # Ensure that both training data for given group has controls
                assert (sum(group_HC_train_mask) > 0)

                group_scaler = StandardScaler()
                group_scaler.fit(X_train[group_HC_train_mask])
                X_train[group_train_mask] = group_scaler.transform(X_train[group_train_mask])

                # And apply to test data (if site is available)
                if (sum(group_test_mask) > 0):
                    X_test[group_test_mask] = group_scaler.transform(X_test[group_test_mask])

        elif CV == 'loso':

            for i_g, group in enumerate(np.unique(groups_train)):
                # Create masks for patients and controls for training and test data for given group
                group_HC_train_mask = (groups_train == group) & (y_train == 0)
                group_train_mask = (groups_train == group)

                # Ensure that both training data for given group has controls
                assert (sum(group_HC_train_mask) > 0)

                group_scaler = StandardScaler()
                group_scaler.fit(X_train[group_HC_train_mask])
                X_train[group_train_mask] = group_scaler.transform(X_train[group_train_mask])

            # Fit scaler on HC from left out site & apply to entire left out site
            group_scaler = StandardScaler()
            group_scaler.fit(X_test[y_test == 0])
            X_test = group_scaler.transform(X_test)

    return X_train, X_test


def run_binary_clf(analysis_dir, WG_df, FS_cols, class_label, N_threshold_c0, N_threshold_c1,
                   outer_cv_folds, outer_cv_repeats, n_undersamplings,
                   CV='kfold', permutations=0):
    
    # Only use samples with given thresholds per class
    mask, counts_df = has_N_per_class(data_df=WG_df, class_label=class_label,
                                      N_threshold_c0=N_threshold_c0,
                                      N_threshold_c1=N_threshold_c1,
                                      verbose=False)
    WG_df = WG_df.loc[mask]

    # Extract X, y and groups
    X = WG_df[FS_cols].to_numpy()
    y = WG_df[class_label].to_numpy()
    groups = WG_df['MultiSiteID'].to_numpy()

    # Permute labels within sites (only when running permutations)
    if permutations == 0:
        clf_results_path = os.path.join(analysis_dir, 'clf_results.npz')
    else:
        rng = np.random.default_rng(permutations)
        y_perm = y.copy()
        y_perm[:] = np.nan
        for group in np.unique(groups):
            mask = group == groups
            y_perm[mask] = rng.permutation(y[mask])
        y = y_perm
        clf_results_path = os.path.join(analysis_dir, 'clf_results_perm_' + str(permutations - 1).zfill(4) + '.npz')

    # Skip classification if output already exists, if so, skip
    if os.path.exists(clf_results_path):
        if permutations == 0:
            print("Found classification results, skipping...")
        return

        # Set up CV with either Stratified K-Fold or Leave-One-Site-Out
    if CV == 'kfold':
        cv_splits = repeated_group_stratified_KFold_splits(y=y, groups=groups,
                                                           n_folds=outer_cv_folds, n_repeats=outer_cv_repeats)
    elif CV == 'loso':
        cv_splits = leave_one_site_out_splits(y=y, groups=groups)

    n_splits = len(cv_splits)

    # Set up scorer and arrays to store results
    scoring = Evaluater()
    metric_labels = scoring.evaluate_labels()
    metric_scores = np.zeros((n_splits, len(metric_labels)))
    predictions = np.ones((len(y), n_splits)) * -1
    scores = np.ones((len(y), n_splits)) * -1

    # Print N_features, N_classes and N_sites
    y_vals, y_counts = np.unique(y, return_counts=True)
    n_sites = len(np.unique(groups))
    if permutations == 0:
        print(f"{y_counts[y_vals == 0]} HC and {y_counts[y_vals == 1]} patients from {n_sites} sites")
        print(f"Features shape: {X.shape}")

    # Run ML
    time1 = time()

    for i_split, (train_idx, test_idx) in enumerate(tqdm(cv_splits)):

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        # 1. Apply site-wise scaling using HC
        X_train, X_test = site_wise_scaling(X_train, y_train, groups_train,
                                            X_test, y_test, groups_test,
                                            groups, CV=CV)

        # 2. Impute missing values for both training and test set
        tmp_imputer = SimpleImputer(strategy='mean')
        X_train = tmp_imputer.fit_transform(X_train)
        X_test = tmp_imputer.transform(X_test)

        # 3. Store decision values across undersamplings
        dec_vals_test = np.zeros((n_undersamplings, len(test_idx)))
        dec_vals_full_train = np.zeros((n_undersamplings, len(train_idx)))

        rus_idx_total = []

        # 4. Apply site-wise undersampling on training data, do this 10 times!
        for i_us in range(n_undersamplings):
            rus_idx = SiteWise_RandomUnderSampler(groups_train, y_train, random_state=i_us)
            X_train_ = X_train[rus_idx]
            y_train_ = y_train[rus_idx]
            groups_train_ = groups_train[rus_idx]

            rus_idx_total.append(rus_idx)

            # 5. Set up SVM classifier and fit on undersampled data
            clf = SVC(kernel='linear', class_weight='balanced', C=1)
            clf.fit(X_train_, y_train_)

            # 6. Retrieve SVM decision value scores for test data
            dec_vals_test[i_us, :] = clf.decision_function(X_test)

        # 7. Take median of decision scores across undersamplings
        y_score = np.median(dec_vals_test, axis=0)

        # 8. Transform decision values into predicted classes
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > 0] = 1

        # 9. Store predictions, decision values and classification metrics
        predictions[test_idx, i_split] = y_pred
        scores[test_idx, i_split] = y_score
        metric_scores[i_split, :] = scoring.evaluate_prediction(y_score=y_score,
                                                                y_pred=y_pred,
                                                                y_true=y_test)

    time2 = time()

    if permutations == 0:
        print()
        print(f"Finished classification in {(time2 - time1) / 60:.3f} minutes. Obtained performance:\n")
        for m_idx, m_label in enumerate(metric_labels):
            print("mean {}: {:.3f}, std: {:.3f}".format(m_label,
                                                        metric_scores[:, m_idx].mean(),
                                                        metric_scores[:, m_idx].std()))
        print()

    # Store clf dataframe and sample size information (not for permutations)
    if permutations == 0:
        WG_df.to_csv(os.path.join(analysis_dir, 'clf_data.csv'))
        counts_df.to_csv(os.path.join(analysis_dir, 'clf_samples_sizes.csv'))
        if 'cross_disorder' in clf_results_path:
            WG_df.groupby(['WG', class_label]).size().to_csv(os.path.join(analysis_dir, 'clf_WG_sizes.csv'))

            # Store clf results
    np.savez(file=clf_results_path,
             cv_splits=np.array(cv_splits, dtype=object),
             predictions=predictions,
             scores=scores,
             metric_labels=np.array(metric_labels),
             metric_scores=metric_scores,
             dtype=object)


def run_binary_clf_wo_scaling(analysis_dir, WG_df, FS_cols, class_label,
                              outer_cv_folds, outer_cv_repeats, n_undersamplings,
                              CV='kfold', permutations=0):
    # Parse features, classes and groups
    X = WG_df[FS_cols].to_numpy()
    y = WG_df[class_label].to_numpy(dtype=float)
    groups = WG_df['MultiSiteID'].to_numpy()

    # Permute labels within sites (only when running permutations)
    if permutations == 0:
        clf_results_path = os.path.join(analysis_dir, 'clf_results.npz')
    else:
        rng = np.random.default_rng(permutations)
        y_perm = y.copy()
        y_perm[:] = np.nan
        for group in np.unique(groups):
            mask = group == groups
            y_perm[mask] = rng.permutation(y[mask])
        y = y_perm
        clf_results_path = os.path.join(analysis_dir, 'clf_results_perm_' + str(permutations - 1).zfill(4) + '.npz')

    # Skip classification if output already exists
    if os.path.exists(clf_results_path):
        if permutations == 0:
            print("Found classification results, skipping...")
        return

        # Set up CV with either Stratified K-Fold or Leave-One-Site-Out
    if CV == 'kfold':
        cv_splits = repeated_group_stratified_KFold_splits(y=y, groups=groups,
                                                           n_folds=outer_cv_folds, n_repeats=outer_cv_repeats)
    elif CV == 'loso':
        cv_splits = leave_one_site_out_splits(y=y, groups=groups)

    n_splits = len(cv_splits)

    # Set up scorer and arrays to store results
    scoring = Evaluater()
    metric_labels = scoring.evaluate_labels()
    metric_scores = np.zeros((n_splits, len(metric_labels)))
    predictions = np.ones((len(y), n_splits)) * -1
    scores = np.ones((len(y), n_splits)) * -1

    # Print N_features, N_classes and N_sites
    y_vals, y_counts = np.unique(y, return_counts=True)
    n_sites = len(np.unique(groups))
    if permutations == 0:
        print(
            f"N {class_label}==0 {y_counts[y_vals == 0]}, N {class_label}==1 {y_counts[y_vals == 1]}, {n_sites} sites")
        print(f"Features shape: {X.shape}")

    # Run ML
    time1 = time()

    for i_split, (train_idx, test_idx) in enumerate(tqdm(cv_splits)):

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        # 1. Impute missing values for both training and test set
        tmp_imputer = SimpleImputer(strategy='mean')
        X_train = tmp_imputer.fit_transform(X_train)
        X_test = tmp_imputer.transform(X_test)

        # 2. Store decision values across undersamplings
        dec_vals_test = np.zeros((n_undersamplings, len(test_idx)))
        dec_vals_full_train = np.zeros((n_undersamplings, len(train_idx)))

        rus_idx_total = []

        # 3. Apply site-wise undersampling on training data, do this 10 times!
        for i_us in range(n_undersamplings):
            rus_idx = SiteWise_RandomUnderSampler(groups_train, y_train, random_state=i_us)
            X_train_ = X_train[rus_idx]
            y_train_ = y_train[rus_idx]
            groups_train_ = groups_train[rus_idx]

            rus_idx_total.append(rus_idx)

            # 4. Set up SVM classifier and fit on undersampled data
            clf = SVC(kernel='linear', class_weight='balanced', C=1)
            clf.fit(X_train_, y_train_)

            # 5. Retrieve SVM decision value scores for test data
            dec_vals_test[i_us, :] = clf.decision_function(X_test)

        # 6 Take median of decision scores across undersamplings
        y_score = np.median(dec_vals_test, axis=0)

        # 7. Transform decision values into predicted classes
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > 0] = 1

        # 8. Store predictions, decision values and classification metrics
        predictions[test_idx, i_split] = y_pred
        scores[test_idx, i_split] = y_score
        metric_scores[i_split, :] = scoring.evaluate_prediction(y_score=y_score,
                                                                y_pred=y_pred,
                                                                y_true=y_test)

    time2 = time()

    if permutations == 0:
        print()
        print(f"Finished classification in {(time2 - time1) / 60:.3f} minutes. Obtained performance:\n")
        for m_idx, m_label in enumerate(metric_labels):
            print("mean {}: {:.3f}, std: {:.3f}".format(m_label,
                                                        metric_scores[:, m_idx].mean(),
                                                        metric_scores[:, m_idx].std()))
        print()

    np.savez(file=clf_results_path,
             cv_splits=np.array(cv_splits, dtype=object),
             predictions=predictions,
             scores=scores,
             metric_labels=np.array(metric_labels),
             metric_scores=metric_scores,
             dtype=object)
