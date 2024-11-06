import pickle
import numpy as np
import pandas as pd
from numba import njit
from sklearn.model_selection import StratifiedGroupKFold


def load_habitat_dataset(habitat, mod_label=None, data_dir='resources', model='vggish', feat_time='960'):
    """Returns a dataframe for the given habitat at the given feature time

    Parameters
    ----------
    habitat: str
        The habitat for which to load data
    data_dir: str
        The location of the data files (default = resources)
    model: str
        The model for which data to load (default = vggish)
    feat_time: str
        The feature time in ms (default = 960)

    Returns
    -------
    data
        Dataset for the given habitat and feature time
    """
    if mod_label is None:
        data = pd.read_pickle(f"{data_dir}/{model}_{habitat}_features_{feat_time}ms_pruned.pkl")
    else:
        data = pd.read_pickle(f"{data_dir}/{model}_{habitat}_{mod_label}_{feat_time}ms.pkl")
    return data

def save_habitat_dataset(dataset, habitat, mod_label, data_dir='resources', model='vggish', feat_time='960'):
    """Writes habitat dataset to disk

    Parameters
    ----------
    dataset: pandas dataframe
        The habitat dataset
    habitat: str
        The habitat for which to load data
    mod_label: str
        Label for the dataset (if dataset was modified)
    data_dir: str
        The location of the data files (default = resources)
    model: str
        The model for which data to load (default = vggish)
    feat_time: str
        The feature time in ms (default = 960)
    """
    dataset.to_pickle(f"{data_dir}/{model}_{habitat}_{mod_label}_{feat_time}ms.pkl")

@njit
def time_and_date(date, time):
    """Returns the string concatenation of date and time.

    Parameters
    ----------
    date: str, datetime
        The date
    time: str, datetime
        The time

    Returns
    -------
    string
        The concatenated datetime value
    """
    return str(date) + str(time)

def get_gmm_fold_models(habitat, feat_time='960', fold=0, model_dir='resources/models'):
    """Returns GMM, Scaler, PCA models for a particular fold

    Parameters
    ----------
    habitat: str
        The habitat label
    feat_time: str
        The feature time in ms (default=960)
    fold: int
        The fold number (default = 0)
    model_dir: str
        Path to model directory

    Returns
    -------
    scaler: sklearn model
        Scaler model
    pca: sklearn model
        PCA model
    gmm_h: skelarn model
        GMM model for H land use
    gmm_m: sklearn model
        GMM model for M land use
    gmm_l: sklearn model
        GMM model for L land use
    """
    with open(f'{model_dir}/gmm_models_{habitat}_fold_{fold}_{feat_time}ms_pca_0.95.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    scaler = model_dict['Scaler']
    pca = model_dict['PCA']
    gmm_h = model_dict['GMM H']
    gmm_m = model_dict['GMM M']
    gmm_l = model_dict['GMM L']
    return scaler, pca, gmm_h, gmm_m, gmm_l

def get_gmm_models(habitat, feat_time='960', model_dir='resources/models'):
    """Returns GMM, Scaler, PCA aggregate models for a particular habitat

    Parameters
    ----------
    habitat: str
        The habitat label
    feat_time: str
        The feature time in ms (default=960)
    model_dir: str
        Path to model directory

    Returns
    -------
    scaler: sklearn model
        Scaler model
    pca: sklearn model
        PCA model
    gmm_h: skelarn model
        GMM model for H land use
    gmm_m: sklearn model
        GMM model for M land use
    gmm_l: sklearn model
        GMM model for L land use
    """
    with open(f'{model_dir}/gmm_models_{habitat}_{feat_time}ms_pca_0.95_full.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    scaler = model_dict['Scaler']
    pca = model_dict['PCA']
    gmm_h = model_dict['GMM H']
    gmm_m = model_dict['GMM M']
    gmm_l = model_dict['GMM L']
    return scaler, pca, gmm_h, gmm_m, gmm_l

def extract_features_for_fold(dataset, cv_fold_no=0, train_data=True, no_folds=4, verbose=False):
    """Returns training or testing features in the form of h_data, m_data, l_data from dataset
    for a particular fold using StratifiedGroupKFold method from sklearn.

    Parameters
    ----------
    dataset: pandas dataframe
        The dataset containing audio features, landuse, and sites
    cv_fold_no: int
        The fold number for which to return training features (default = 0)
    train_data: bool
        If true, returns training features, else returns testing features (default = True)
    no_folds: int
        Number of folds for Stratified Group K-fold cross validation (default = 4)
    verbose: bool
        Flag for verbose output (default = False)

    Returns
    -------
    h_data: numpy array
        data for H landuse
    m_data: numpy array
        data for M landuse
    l_data: numpy array
        data for L landuse
    """
    Y = np.asarray(list(dataset['landuse']))
    GROUPS = dataset['sites']
    CV = StratifiedGroupKFold(n_splits=no_folds)
    for i, (train_index, test_index) in enumerate(CV.split(dataset, Y, GROUPS)):
        if i == cv_fold_no:
            if verbose:
                print(f"Fold {i}")
                if train_data:
                    print(f"Training group={GROUPS[train_index].unique()}")
                else:
                    print(f"Testing group={GROUPS[test_index].unique()}")

            indices = train_index
            if not train_data:
                indices = test_index
       
            h_data = np.asarray(list(dataset.iloc[indices].loc[dataset['landuse'] == 'H']['feats']))
            assert h_data.shape[0] == GROUPS[indices].str.count('H').sum()

            m_data = np.asarray(list(dataset.iloc[indices].loc[dataset['landuse'] == 'M']['feats']))
            assert m_data.shape[0] == GROUPS[indices].str.count('M').sum()

            l_data = np.asarray(list(dataset.iloc[indices].loc[dataset['landuse'] == 'L']['feats']))
            assert l_data.shape[0] == GROUPS[indices].str.count('L').sum()

            return h_data, m_data, l_data
        
def extract_landuse_features(dataset):
    """Returns audio features segregated by landuse for given dataset

    Parameters
    ----------
    dataset: pandas dataframe
        The dataset for which to segregate features

    Returns
    -------
    H_train: numpy array
        Features for H landuse
    M_train: numpy array
        Features for M landuse
    L_train: numpy array
        Features for L landuse
    """
    h_data = dataset[dataset['landuse'] == 'H']
    H_train = np.asarray(list(h_data['feats']))
    m_data = dataset[dataset['landuse'] == 'M']
    M_train = np.asarray(list(m_data['feats']))
    l_data = dataset[dataset['landuse'] == 'L']
    L_train = np.asarray(list(l_data['feats']))

    return H_train, M_train, L_train