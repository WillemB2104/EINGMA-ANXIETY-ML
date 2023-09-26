import datetime
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from tqdm.notebook import tqdm

from .evaluation_classifier import Evaluater


def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def string_to_dict(string, pattern):
    regex = re.sub(r'{(.+?)}', r'(?P<_\1>.+)', pattern)
    values = list(re.search(regex, string).groups())
    keys = re.findall(r'{(.+?)}', pattern)
    _dict = dict(zip(keys, values))
    return _dict


def reorder_df_col(df, column_to_insert, column_before_insert):
    df.insert(df.columns.get_loc(column_before_insert), column_to_insert, df.pop(column_to_insert))


def load_most_recent_WG_data(WG):
    if WG == 'PD':
        working_group_dir = '/data/wbbruin/Documents/Stage-Machine-Learning/ENIGMA_PANIC/POOLED/'
    elif WG == 'SAD':
        working_group_dir = '/data/wbbruin/Documents/Stage-Machine-Learning/ENIGMA_SAD/'
    elif WG == 'GAD':
        working_group_dir = '/data/wbbruin/Documents/Stage-Machine-Learning/ENIGMA_GAD/'

    files = glob(os.path.join(working_group_dir, '*POOLED_DATA_FOR_CROSS_DISORDER.csv'))

    if len(files) > 0:
        print("Loading most recent pooled data...")
        most_recent_idx = np.argmax([datetime.datetime.strptime(
            f.split('_POOLED_DATA_FOR_CROSS_DISORDER')[0].split('_')[-1], "%Y-%m-%d") for f in files])
        print("Found data stored @ {}".format(files[most_recent_idx]))
        df = pd.read_csv(files[most_recent_idx], low_memory=False)

        return df
    else:
        print("No data found!")


def merge_bool_masks(*masks):
    return np.array([all(tup) for tup in zip(*masks)])


def has_N_per_class(data_df, class_label, subject_mask=None, N_threshold_c0=10, N_threshold_c1=1, verbose=True):
    groups = np.array(data_df.MultiSiteID.values)

    if subject_mask is not None:
        tmp_df = data_df.loc[subject_mask, ['SubjID', class_label, 'MultiSiteID']].copy()
    else:
        tmp_df = data_df.loc[:, ['SubjID', class_label, 'MultiSiteID']].copy()

    all_sites = np.unique(tmp_df.MultiSiteID)
    unique_class_values = sorted(np.unique(tmp_df[class_label]))
    assert len(unique_class_values) == 2

    # Store number of included subjects per class and site in Dataframe
    counts_df = tmp_df.groupby(['MultiSiteID', class_label]).size()
    counts_df = pd.DataFrame(counts_df)
    counts_df = counts_df.reset_index()
    counts_df = counts_df.pivot(index='MultiSiteID', columns=class_label)
    counts_df.columns = counts_df.columns.droplevel(0)
    counts_df.columns.name = None
    counts_df = counts_df.reset_index()

    if N_threshold_c0 + N_threshold_c1 > 0:
        included_sites = counts_df.loc[(counts_df[unique_class_values[0]] >= N_threshold_c0) &
                                       (counts_df[unique_class_values[1]] >= N_threshold_c1)]['MultiSiteID'].values
        excluded_sites = list(set(all_sites).difference(included_sites))
    else:
        included_sites = all_sites
        excluded_sites = []

    counts_df = counts_df.loc[counts_df.MultiSiteID.isin(included_sites)]
    site_mask = [g not in excluded_sites for g in groups]

    if verbose:
        N_excluded = len(tmp_df.loc[tmp_df.MultiSiteID.isin(excluded_sites)])
        print(f"Excluded {N_excluded} subjects belonging to {len(excluded_sites)} different sites:")
        excluded_sites_str = '\n'.join(excluded_sites)
        print(f"{excluded_sites_str}")
    return site_mask, counts_df


def exclude_subjects_with_missing_features(df, FS_cols, completeness_threshold=0.75):
    # Extract FS features
    X = df[FS_cols].values

    N_features = len(FS_cols)

    # Create mask for subjects that have too many missing values
    N_missing_per_subject = np.sum(np.isnan(X), axis=1)
    p_missing_per_subject = N_missing_per_subject / float(N_features)
    p_missing_inclusion_mask = (p_missing_per_subject < (1 - completeness_threshold))
    n_missing_excluded = sum(~p_missing_inclusion_mask)

    print(f"{sum(N_missing_per_subject > 0)} of {len(N_missing_per_subject)} subjects have >=1 missing features")
    print(f"{n_missing_excluded} subjects excluded with >{int((1 - completeness_threshold) * 100)}% missing features")
    print(df.loc[~p_missing_inclusion_mask].groupby(['WG', 'Dx']).size())
    print()

    df = df.loc[p_missing_inclusion_mask]

    return df


def extract_FS_cols(SPREADSHEET_TEMPLATES_DIR):
    spreadsheet_files = os.listdir(SPREADSHEET_TEMPLATES_DIR)
    spreadsheet_columns = {}

    for f in spreadsheet_files:

        if '.csv' in f:
            df = pd.read_csv(os.path.join(SPREADSHEET_TEMPLATES_DIR, f))
        else:
            df = pd.read_excel(os.path.join(SPREADSHEET_TEMPLATES_DIR, f), header=1)

        spreadsheet_columns[f] = df.columns.values

    # Remove columns that are duplicated across sheets
    for f in spreadsheet_files:

        if f != 'CorticalMeasuresENIGMA_SurfAvg.csv':

            if ('LSurfArea' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'LSurfArea')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

            if ('RSurfArea' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'RSurfArea')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

        if f != 'CorticalMeasuresENIGMA_ThickAvg.csv':

            if ('LThickness' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'LThickness')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

            if ('RThickness' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'RThickness')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

        if f != 'LandRvolumes.csv':

            if ('ICV' in spreadsheet_columns[f]):
                idx_to_del = np.where(spreadsheet_columns[f] == 'ICV')[0]
                spreadsheet_columns[f] = np.delete(spreadsheet_columns[f], idx_to_del)

    # Extract FreeSurfer labels
    FS_cols = np.concatenate([
        spreadsheet_columns['CorticalMeasuresENIGMA_SurfAvg.csv'][1:],
        spreadsheet_columns['CorticalMeasuresENIGMA_ThickAvg.csv'][1:],
        spreadsheet_columns['LandRvolumes.csv'][1:]
    ])

    print("Total FS columns: {}".format(len(FS_cols)))

    # Create a subset without global features (i.e. summarized measures over hemipsheres and ICV)
    global_FS_features = ['LSurfArea', 'RSurfArea', 'LThickness', 'RThickness', 'ICV']
    subset_mask = [f not in global_FS_features for f in FS_cols]
    FS_cols_wo_global = FS_cols[subset_mask]

    print("Total FS columns without global hemishpere measures and ICV: {}".format(len(FS_cols_wo_global)))

    # Parse out different modalities (CT/CSA/SUBVOL)
    ct_mask = ['thick' in f for f in FS_cols_wo_global]
    csa_mask = ['surf' in f for f in FS_cols_wo_global]
    subcort_mask = ~np.array(ct_mask) & ~np.array(csa_mask)

    assert sum(ct_mask) + sum(csa_mask) + sum(subcort_mask) == len(FS_cols_wo_global)

    return FS_cols, FS_cols_wo_global


def create_clf_results_overview_table(result_paths, classification_dir, filename='clf_results.csv', N_PERMS=1000):

    # Optionally add 'positive_predictive_value', 'negative_predictive_value'
    clf_metrics_to_report = ['AUC', 'balanced_accuracy', 'sensitivity', 'specificity']
    sample_sizes_to_report = ['N_class_0', 'N_class_1', 'N_sites']

    pooled_results_columns = np.concatenate((['analysis_label', 'filter_label'], clf_metrics_to_report,
                                             ['p-value'], ['AUC perms'], sample_sizes_to_report))
    pooled_results_df = pd.DataFrame(columns=pooled_results_columns)

    # Iterate over results
    for i_r, result_path in enumerate(tqdm(result_paths)):

        filter_label = ''
        analysis_label = result_path.split('results/')[-1].split('/')[1]
        if '_subanalysis_' in analysis_label:
            filter_label = analysis_label.split('_subanalysis_')[1]
            analysis_label = analysis_label.split('_subanalysis_')[0]

        # Extract classification performances
        clf_metric_scores = np.load(result_path, allow_pickle=True)['metric_scores']
        clf_metric_labels = np.load(result_path, allow_pickle=True)['metric_labels']
        clf_metrics_formatted = [
            f'{clf_metric_scores[:, clf_metric_labels == l].mean():.2f} ({clf_metric_scores[:, clf_metric_labels == l].std():.2f})'
            for l in clf_metrics_to_report]

        # Extract sample sizes
        sample_sizes_df = pd.read_csv(result_path.replace('clf_results.npz', 'clf_samples_sizes.csv'))
        N_class_0, N_class_1 = int(sample_sizes_df['0.0'].sum()), int(sample_sizes_df['1.0'].sum())
        N_sites = len(sample_sizes_df.MultiSiteID.unique())
        sample_sizes_formatted = [N_class_0, N_class_1, N_sites]

        # Calculate p-value for AUC using permutatations
        perm_dir = result_path.replace('clf_results.npz', 'permutations')
        perm_results = sorted(glob(os.path.join(perm_dir, '*.npz')))
        perm_results = [np.load(p)['metric_scores'] for p in perm_results]
        perm_results = np.array(perm_results)
        assert perm_results.shape[0] == N_PERMS

        neutral_AUC = clf_metric_scores[:, np.where(np.array(clf_metric_labels) == 'AUC')[0][0]].mean()
        perm_AUCs = perm_results[:, :, np.where(np.array(clf_metric_labels) == 'AUC')[0][0]]
        perm_AUCs = perm_AUCs.mean(axis=1)
        p_val_unc = (np.sum(perm_AUCs >= neutral_AUC) + 1.0) / (len(perm_AUCs) + 1)
        p_val_unc = f'{p_val_unc:.5f}'

        perm_AUCs_mean = perm_AUCs.mean()

        tmp_df = pd.DataFrame(columns=pooled_results_columns,
                              data=np.c_[[analysis_label], [filter_label],
                                         [clf_metrics_formatted], [p_val_unc], [perm_AUCs_mean],
                                         [sample_sizes_formatted]])

        pooled_results_df = pd.concat([pooled_results_df, tmp_df])

    # Add asterix to string for significant p-values
    sig_results_mask = pooled_results_df['p-value'].astype(float) < 0.05
    pooled_results_df.loc[sig_results_mask, 'AUC'] = pooled_results_df.loc[sig_results_mask, 'AUC'] + '*'

    # Store overview
    pooled_results_df.to_csv(os.path.join(classification_dir, filename))

    return pooled_results_df


def autolabel(ax, positions, heights, significance_mask, fontsize_L2):
    """
    Attach a text label above each bar displaying its height
    """
    for pos, height, sig in zip(positions, heights, significance_mask):
        if sig > 0:
            ax.text(pos,
                    0.85,  # height + 0.01,
                    '*',
                    fontsize=fontsize_L2 * 1.5, fontweight='heavy',
                    ha='center', va='center')


def plot_binary_clf_results_on_ax(ax, clf_tasks, clf_tasks_labels,
                                  BINARY_CLF_DIR, N_PERMS, outer_cv_folds, outer_cv_repeats):

    metrics_of_interest = ['AUC', 'balanced_accuracy', 'sensitivity', 'specificity']

    scoring = Evaluater()
    metric_labels = scoring.evaluate_labels()
    metrics_of_interest_indices = [np.where(np.array(metric_labels) == mi)[0][0] for mi in metrics_of_interest]
    colors = sns.cubehelix_palette(len(metrics_of_interest), start=0.8, rot=-1.2, light=0.9, dark=0.3)

    # Load binary classification results
    metrics_per_clf_task = np.zeros((len(clf_tasks), outer_cv_folds * outer_cv_repeats, len(metric_labels)))
    is_significant = []

    for i, clf_task in enumerate(clf_tasks):
        results_path = os.path.join(BINARY_CLF_DIR, f'{clf_task}', 'clf_results.npz')
        metrics_per_clf_task[i, :, :] = np.load(results_path, allow_pickle=True)['metric_scores']

        # Calculate significance
        neutral_AUC = metrics_per_clf_task[i, :, metrics_of_interest_indices[metrics_of_interest == 'AUC']].mean()

        perm_dir = results_path.replace('clf_results.npz', 'permutations')
        perm_results = sorted(glob(os.path.join(perm_dir, '*.npz')))
        perm_results = [np.load(p)['metric_scores'] for p in perm_results]
        perm_results = np.array(perm_results)
        assert perm_results.shape[0] == N_PERMS

        perm_AUCs = perm_results[:, :, metrics_of_interest_indices[metrics_of_interest == 'AUC']].mean(axis=1)
        p_val = (np.sum(perm_AUCs >= neutral_AUC) + 1.0) / (len(perm_AUCs) + 1)
        is_significant.append(p_val < 0.05)

    plt.style.use('default')
    scale = 1.5
    error_scale = scale * (7 / 3.)

    space = 0.2
    width = (1 - space) / (len(metrics_of_interest))

    fontsize_L1 = 6 * scale
    fontsize_L2 = 5 * scale
    fontdict_labels = {'fontsize': fontsize_L1}

    positions = []
    high_caps = []

    for i, i_metric in enumerate(metrics_of_interest_indices):

        mean = metrics_per_clf_task[:, :, i_metric].mean(axis=-1)
        pos = [j - (1 - space) / 2. + i * width for j in range(1, len(clf_tasks) + 1)]

        tmp = ax.boxplot(metrics_per_clf_task[:, :, i_metric].T,
                         positions=pos,
                         widths=[width] * (len(pos)),
                         patch_artist=True,
                         medianprops={'linewidth': 0.75},
                         flierprops={'markersize': 2.5,
                                     'markeredgewidth': 0.75},
                         meanprops={"marker": "d",
                                    "markerfacecolor": "yellow",
                                    "markeredgecolor": "black",
                                    "markersize": 4},
                         showmeans=True)

        for box in tmp['boxes']:
            box.set(facecolor=colors[i])

        positions = np.append(positions, pos)
        tmp_caps = [t.get_ydata()[0] for t in tmp['caps'][1::2]]
        high_caps = np.append(high_caps, tmp_caps)

    ax.set_xticks([j - (1 - space) / 2. + 1 * width for j in range(1, len(clf_tasks) + 1)])
    ax.set_xticklabels(clf_tasks_labels, fontdict={'fontsize': fontsize_L2}, )
    ax.axhline(linewidth=1, y=0.5, color='r', linestyle='--', alpha=0.5)

    y_min = 0.2
    y_max = 1.0
    y_steps = 0.1

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + y_steps, y_steps))
    ax.set_yticklabels(np.round(np.arange(y_min, y_max + y_steps, y_steps, ), 2), fontdict={'fontsize': fontsize_L2})

    metrics_of_interest_prettified = ['ROC-AUC', 'Balanced Accuracy', 'Sensitivity', 'Specificity']

    # Add significance asterisks (only for AUC)
    significance_mask = np.ravel([np.append(s, ([False] * (len(clf_tasks) - 1))) for s in is_significant]
                                 ).astype(float)
    autolabel(ax, positions, high_caps, significance_mask, fontsize_L2)

    # Add a legend
    legend_elements = [Patch(facecolor=colors[i], label=metrics_of_interest_prettified[i],
                             edgecolor='black', linewidth=1.0) for i in range(len(metrics_of_interest))]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=fontsize_L2, frameon=False)

    sns.despine(ax=ax, top=True, right=True)