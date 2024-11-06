import pickle
import argparse
import numpy as np
from helpers import extract_landuse_features, load_habitat_dataset, save_habitat_dataset, time_and_date
from analysis_libs import get_landuse_threshold, get_likelihood_scores_for_samples, get_predictions_from_scores, run_pca_for_gmm

@np.vectorize
def create_filename(site, date, time):
    """Returns the fully formed filename.

    Parameters
    ----------
    site: str
        The site
    date: str, datetime
        The date
    time: str, datetime
        The time

    Returns
    -------
    string
        The filename
    """
    return f"{site}/{date}_{time}.wav"

def add_component(feats, model):
    """Returns array of GMM component numbers.

    Parameters
    ----------
    feats: ndarray
        Feature vectors
    model: str
        Model shorthand (H, M, or L)

    Returns
    -------
    ndarray
        Array containing GMM component numbers
    """
    res = np.empty(model.shape)
    for i in range(len(model)):
        feat = feats[i].reshape(1, -1)
        if model[i] == 'H':
            res[i] = gmm_h.predict(feat)[0]
        elif model[i] == 'M':
            res[i] = gmm_m.predict(feat)[0]
        elif model[i] == 'L':
            res[i] = gmm_l.predict(feat)[0]
        else:
            res[i] = np.nan
    return res

def add_component_density(feats, model):
    """Returns array of GMM component density.

    Parameters
    ----------
    feats: ndarray
        Feature vectors
    model: str
        Model shorthand (H, M, or L)

    Returns
    -------
    ndarray
        Array containing GMM component density
    """
    res = np.empty(model.shape)
    for i in range(len(model)):
        feat = feats[i].reshape(1, -1)
        if model[i] == 'H':
            res[i] = np.max(gmm_h.predict_proba(feat)[0])
        elif model[i] == 'M':
            res[i] = np.max(gmm_m.predict_proba(feat)[0])
        elif model[i] == 'L':
            res[i] = np.max(gmm_l.predict_proba(feat)[0])
        else:
            res[i] = np.nan
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ha', '--habitat',
                            default='grassland', help='Specify the habitat (default = grassland)')
    parser.add_argument('-o', '--output_dir', type=str, default='resources/mod_data',
                            help="Path to save dataframe (default = resources/mod_data)")
    parser.add_argument('-v', '--verbose', action='store_true',
                            help="Flag to make the program verbose")

    args = parser.parse_args()

    HABITAT = args.habitat
    FEAT_TIME = '960'
    VAR_THRESHOLD = 0.95
    INPUT_DIR = 'resources/models'
    OUTPUT_DIR = args.output_dir
    VERBOSE = args.verbose
    if HABITAT == 'grassland':
        OP_THRESHOLD = 76
    elif HABITAT == 'deciduous':
        OP_THRESHOLD = 64
    elif HABITAT == 'scrub':
        OP_THRESHOLD = 22
    elif HABITAT == 'evergreen':
        OP_THRESHOLD = 43

    if VERBOSE:
        print(f"Loading dataset {HABITAT} {FEAT_TIME}ms")
    dataset = load_habitat_dataset(habitat=HABITAT, feat_time=FEAT_TIME)

    if VERBOSE:
        print("Generating dataframes for the full model with no threshold and optimal threshold")

    # loading models
    with open(f'{INPUT_DIR}/gmm_models_{HABITAT}_{FEAT_TIME}ms_pca_{VAR_THRESHOLD}_full.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    pca = model_dict['PCA']
    scaler = model_dict['Scaler']
    gmm_h = model_dict['GMM H']
    gmm_m = model_dict['GMM M']
    gmm_l = model_dict['GMM L']

    # loading data
    X_test = np.asarray(list(dataset['feats']))
    h_test, m_test, l_test = extract_landuse_features(dataset)
    h_test_pca = run_pca_for_gmm(h_test, pca, scaler)
    m_test_pca = run_pca_for_gmm(m_test, pca, scaler)
    l_test_pca = run_pca_for_gmm(l_test, pca, scaler)

    if VERBOSE:
        print("Generating predictions")
    # generating pca features, h m l scores, predictions
    X_test_pca = run_pca_for_gmm(X_test, pca, scaler)
    h_score, m_score, l_score = get_likelihood_scores_for_samples(X_test_pca, gmm_h, gmm_m, gmm_l)
    predictions = get_predictions_from_scores(h_score, m_score, l_score)
    h_thres = get_landuse_threshold(h_test_pca, gmm_h, gmm_m, gmm_l, true_label='H', ptile=OP_THRESHOLD)
    m_thres = get_landuse_threshold(m_test_pca, gmm_h, gmm_m, gmm_l, true_label='M', ptile=OP_THRESHOLD)
    l_thres = get_landuse_threshold(l_test_pca, gmm_h, gmm_m, gmm_l, true_label='L', ptile=OP_THRESHOLD)
    thresholds = [h_thres, m_thres, l_thres]

    dataset['pca_feats'] = list(X_test_pca)
    dataset['datetime'] = dataset.apply(lambda x: time_and_date(x.date, x.time), axis=1)
    dataset['H score'] = h_score
    dataset['M score'] = m_score
    dataset['L score'] = l_score
    dataset['predicted'] = predictions

    thresholded_predictions = get_predictions_from_scores(h_score, m_score, l_score, thresholds=thresholds)
    dataset['thres_pred'] = thresholded_predictions

    if VERBOSE:
        print("Generating offsets")
    # add offset for audio file
    dataset['offset'] = dataset.groupby(by=['datetime', 'sites'], as_index=False).cumcount()
    
    if VERBOSE:
        print("Unique offset values:")
        print(dataset['offset'].unique())

    if VERBOSE:
        print("Adding GMM component info")

    dataset['component'] = add_component(dataset['pca_feats'], dataset['predicted'])
    dataset['component_density'] = add_component_density(dataset['pca_feats'], dataset['predicted'])
    dataset['thres_component'] = add_component(dataset['pca_feats'], dataset['thres_pred'])
    dataset['thres_component_density'] = add_component_density(dataset['pca_feats'], dataset['thres_pred'])

    if VERBOSE:
        print("Adding filename column")

    dataset['filename'] = create_filename(dataset.sites, dataset.date, dataset.time)

    if VERBOSE:
        print(dataset.head())
        print(dataset.info())
        print("Saving dataset")
    
    save_habitat_dataset(dataset, HABITAT, 'full_predicted_pca_0.95', OUTPUT_DIR)

    if VERBOSE:
        print("Testing saved dataframe")
        df = load_habitat_dataset(HABITAT, 'full_predicted_pca_0.95', data_dir=OUTPUT_DIR)
        print(df.head())
        print(df.info())