import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from helpers import load_habitat_dataset, extract_features_for_fold, get_gmm_fold_models
from analysis_libs import get_likelihood_scores_for_samples, get_true_positives, run_pca_for_gmm

def run_pca_for_row(row_data, scaler, pca):
    """Returns PCA transformed feature vectors for a dataframe row.

    Parameters
    ----------
    row_data: ndarray
        Feature vector for the row
    scaler: sklearn model
        Scaler model
    pca: sklearn model
        PCA model

    Returns
    -------
    ndarray
        PCA transformed feature vectors
    """
    data = np.asarray(list(row_data))
    reshaped_data = data.reshape(1, -1)
    transformed_data = pca.transform(scaler.transform(reshaped_data))[0]
    return transformed_data

def run_gmm_for_dataset(dataset, no_folds, fold_no, scaler, pca, gmm_h, gmm_m, gmm_l, verbose):
    """Run GMM for dataset and generate dataframe with predicted scores for fold.

    Parameters
    ----------
    dataset: pandas dataframe
        Habitat dataset
    no_folds: int
        Total number of folds
    fold_no: int
        Current fold number
    scaler: sklearn model
        Scaler model
    pca: sklearn model
        PCA model
    gmm_h: sklearn model
        GMM model for H land use
    gmm_m: sklearn model
        GMM model for M land use
    gmm_l: sklearn model
        GMM model for L land use
    verbose: bool
        Set verbosity of the model

    Returns
    -------
    pandas dataframe
        Dataframe containing predicted GMM scores for current fold
    """
    Y = np.asarray(list(dataset['landuse']))
    GROUPS = dataset['sites']
    CV = StratifiedGroupKFold(n_splits=no_folds)
    for i, (train_index, test_index) in enumerate(CV.split(dataset, Y, GROUPS)):
        if i == fold_no:
            if verbose:
                print(f"Fold {i}")
                print(f"Testing group={GROUPS[test_index].unique()}")

            df = dataset.iloc[test_index]
            res = df['feats'].apply(run_pca_for_row, scaler=scaler, pca=pca)
            df = df.assign(pca_feats=res.values)
            h_scores, m_scores, l_scores = get_likelihood_scores_for_samples(np.asarray(list(df['pca_feats'])), gmm_h, gmm_m, gmm_l)
            df = df.assign(hscores=h_scores)
            df = df.assign(mscores=m_scores)
            df = df.assign(lscores=l_scores)
            df = df.assign(fold=i)
            return df

def generate_thresholds_for_fold(dataset, habitat, feat_time, no_folds, thresholds_list, verbose):
    """Run GMM for dataset and generate fold-wise absolute threshold values.

    Parameters
    ----------
    dataset: pandas dataframe
        Habitat dataset
    habitat: str
        Habitat name
    feat_time: str
        Feature time to use for threshold generation
    no_folds: int
        Total number of folds
    thresholds_list: list
        list of percentile thresholds
    verbose: bool
        Set verbosity of the model

    Returns
    -------
    dict
        Dictionary of fold-wise absolute threshold values
    """
    all_thresholds = {}
    for i in range(no_folds):
        if verbose:
            print(f"Getting thresholds for fold {i}")
        # get models and data
        scaler, pca, gmm_h, gmm_m, gmm_l = get_gmm_fold_models(habitat, feat_time, fold=i)
        h_train, m_train, l_train = extract_features_for_fold(dataset, cv_fold_no=i, train_data=True, no_folds=no_folds, verbose=verbose)
        h_train = run_pca_for_gmm(h_train, pca, scaler)
        m_train = run_pca_for_gmm(m_train, pca, scaler)
        l_train = run_pca_for_gmm(l_train, pca, scaler)

        h_train_true_pos = get_true_positives(h_train, gmm_h, gmm_m, gmm_l, true_label='H')
        m_train_true_pos = get_true_positives(m_train, gmm_h, gmm_m, gmm_l, true_label='M')
        l_train_true_pos = get_true_positives(l_train, gmm_h, gmm_m, gmm_l, true_label='L')   

        for threshold in thresholds_list:
            try:
                x = all_thresholds[threshold]
            except KeyError:
                all_thresholds[threshold] = {}
            if verbose:
                print()
                print(f"Running for threshold {threshold}")
            # get thresholds
            if threshold is not None:
                h_thres = np.percentile(h_train_true_pos[f'H score'], threshold)
                m_thres = np.percentile(m_train_true_pos[f'M score'], threshold)
                l_thres = np.percentile(l_train_true_pos[f'L score'], threshold)
                thresholds = [h_thres, m_thres, l_thres]
            else:
                thresholds = None
        
            if verbose:
                print(f"Thresholds: {thresholds}")

            all_thresholds[threshold][f'fold_{i}'] = thresholds

    return all_thresholds

def generate_gmm_scores(dataset, habitat, feat_time, no_folds, verbose):
    """Run GMM for dataset and generate dataframe with predicted scores.

    Parameters
    ----------
    dataset: pandas dataframe
        Habitat dataset
    habitat: str
        Habitat name
    feat_time: str
        Feature time to use for threshold generation
    no_folds: int
        Total number of folds
    verbose: bool
        Set verbosity of the model

    Returns
    -------
    pandas dataframe
        Dataframe containing all predicted GMM scores
    """
    df = pd.DataFrame()
    for i in range(no_folds):
        # get models and data
        scaler, pca, gmm_h, gmm_m, gmm_l = get_gmm_fold_models(habitat, feat_time, fold=i)

        # run gmm on test data
        fold_df = run_gmm_for_dataset(dataset, no_folds, i, scaler, pca, gmm_h, gmm_m, gmm_l, verbose=verbose)
        df = pd.concat([df, fold_df], ignore_index=True)
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--feat_time', type=str, default='960',
                        help="Specify the feature time in ms (default = 960ms)")
    parser.add_argument('-ha', '--habitat', type=str, default='grassland', 
                        help='Specify the habitat (default = grassland)')
    parser.add_argument('-o', '--output_dir', default='resources/mod_data', type=str,
                        help="The directory for plot output (default = resources/mod_data)")
    parser.add_argument('-th', '--thresholds', action='store_true',
                        help="Set flag to generate thresholds (by default only scores are generated)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Flag to make the program verbose")
    args = parser.parse_args()

    FEAT_TIME = args.feat_time
    HABITAT = args.habitat
    NO_FOLDS = 4
    OUTPUT_DIR = args.output_dir
    VERBOSE = args.verbose
    THRESHOLDS_ONLY = args.thresholds
    THRESHOLDS = [None]
    if THRESHOLDS_ONLY:
        for i in range(1, 100, 3):
            THRESHOLDS.append(i)
    

    dataset = load_habitat_dataset(habitat=HABITAT, feat_time=FEAT_TIME)
        
    # Separately generate scores and thresholds to make program modular
    if not THRESHOLDS_ONLY:
        if VERBOSE:
            print("Generating scores")
        df = generate_gmm_scores(dataset, HABITAT, FEAT_TIME, NO_FOLDS, VERBOSE)
        df.to_pickle(f"{OUTPUT_DIR}/gmm_fold_scores_{HABITAT}_{FEAT_TIME}ms.pkl")
    else:
        all_thresholds = generate_thresholds_for_fold(dataset, HABITAT, FEAT_TIME, NO_FOLDS, THRESHOLDS, VERBOSE)
    if VERBOSE:
        print("Saving data to disk")
    with open(f"{OUTPUT_DIR}/gmm_fold_thresholds_{HABITAT}_{FEAT_TIME}ms.pkl", "wb") as f:
            pickle.dump(all_thresholds, f)

    