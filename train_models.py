import warnings
warnings.simplefilter(action='ignore')

import pickle
import argparse
from helpers import load_habitat_dataset
from argparse import RawTextHelpFormatter
from helpers import extract_landuse_features
from helpers import extract_features_for_fold
from analysis_libs import fit_gmm, fit_pca_for_gmm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-ha', '--habitat',
                            default='grassland', help='Specify the habitat (default = grassland)')
    parser.add_argument('-o', '--output_dir', type=str, default='resources/models',
                            help="Path to save models (default = resources/models)")
    parser.add_argument('-v', '--verbose', action='store_true',
                            help="Flag to make the program verbose")

    args = parser.parse_args()

    HABITAT = args.habitat
    FEAT_TIME = '960'
    CV_FOLDS = 4
    NO_ITERS = 500
    BAYESIAN = True
    VAR_THRESHOLD = 0.95
    GMM_COV_TYPE = 'diag'
    H_COMPS = 100
    M_COMPS = 100
    L_COMPS = 100
    OUTPUT_DIR = args.output_dir
    VERBOSE = args.verbose

    if VERBOSE:
        print(f"Loading dataset {HABITAT} {FEAT_TIME}ms")

    dataset = load_habitat_dataset(habitat=HABITAT, feat_time=FEAT_TIME)

    if VERBOSE:
        print("Training the full model")

    h_train, m_train, l_train = extract_landuse_features(dataset)

    pca_H_train, pca_M_train, pca_L_train, model, scaler = fit_pca_for_gmm(h_train, m_train, l_train, var_threshold=VAR_THRESHOLD, verbose=VERBOSE)

    if VERBOSE:
        print(f"H components: {H_COMPS}, M components: {M_COMPS}, L components: {L_COMPS}")
        
    gmm_h, gmm_m, gmm_l = fit_gmm(pca_H_train, pca_M_train, pca_L_train, H_COMPS, M_COMPS, L_COMPS, BAYESIAN, cov_mat=GMM_COV_TYPE, iters=NO_ITERS, verbose=VERBOSE)

    model_dict = {
        'PCA': model, 'Scaler': scaler,
        'GMM H': gmm_h, 'GMM M': gmm_m, 'GMM L': gmm_l
    }

    if VERBOSE:
        print("Saving models to output directory")

    with open(f'{OUTPUT_DIR}/gmm_models_{HABITAT}_{FEAT_TIME}ms_pca_{VAR_THRESHOLD}_full.pkl', 'wb') as f:
        pickle.dump(model_dict, f)

    if VERBOSE:
        print("Training models for each fold")

    for CURR_FOLD in range(CV_FOLDS):
        if VERBOSE:
            print(f"Running analysis for fold {CURR_FOLD}")
        
        h_train, m_train, l_train = extract_features_for_fold(dataset, cv_fold_no=CURR_FOLD, no_folds=CV_FOLDS, verbose=VERBOSE)

        pca_H_train, pca_M_train, pca_L_train, model, scaler = fit_pca_for_gmm(h_train, m_train, l_train, var_threshold=VAR_THRESHOLD, verbose=VERBOSE)

        if VERBOSE:
            print(f"H components: {H_COMPS}, M components: {M_COMPS}, L components: {L_COMPS}")
        
        gmm_h, gmm_m, gmm_l = fit_gmm(pca_H_train, pca_M_train, pca_L_train, H_COMPS, M_COMPS, L_COMPS, BAYESIAN, cov_mat=GMM_COV_TYPE, iters=NO_ITERS, verbose=VERBOSE)

        model_dict = {
            'PCA': model, 'Scaler': scaler,
            'GMM H': gmm_h, 'GMM M': gmm_m, 'GMM L': gmm_l
        }

        if VERBOSE:
            print("Saving models to output directory")

        with open(f'{OUTPUT_DIR}/gmm_models_{HABITAT}_fold_{CURR_FOLD}_{FEAT_TIME}ms_pca_{VAR_THRESHOLD}.pkl', 'wb') as f:
            pickle.dump(model_dict, f)