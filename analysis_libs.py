import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize
from helpers import extract_features_for_fold, get_gmm_fold_models
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def fit_pca_for_gmm(h_train, m_train, l_train, var_threshold=0.95, verbose=False):
    """Fits PCA to data for dimension reduction before running GMM analysis

    Parameters
    ----------
    h_train: numpy array
        Training samples from H landuse
    m_train: numpy array
        Training samples from M landuse
    l_train: numpy array
        Training samples from L landuse
    var_threshold: float
        Variance threshold for PCA (default = 0.95)
    verbose: bool
        Flag for verbose output (default = False)

    Returns
    -------
    pca_H_train: numpy array
        pca reduced samples for H landuse
    pca_M_train: numpy array
        pca reduced samples for M landuse
    pca_L_train: numpy array
        pca reduced samples for L landuse
    model:
        the fitted pca model
    scaler:
        the fitted scaler used before pca
    """
    start = datetime.now()
    scaler = StandardScaler()
    scaler.fit(np.vstack((h_train, m_train, l_train)))
    scaled_H_train = scaler.transform(h_train)
    scaled_M_train = scaler.transform(m_train)
    scaled_L_train = scaler.transform(l_train)

    if verbose:
        print(f"Original H feature shape: {scaled_H_train.shape}")
        print(f"Original M feature shape: {scaled_M_train.shape}")
        print(f"Original L feature shape: {scaled_L_train.shape}")

    model = PCA(n_components=var_threshold)
    model.fit(np.vstack((scaled_H_train, scaled_M_train, scaled_L_train)))
    pca_H_train = model.transform(scaled_H_train)
    pca_M_train = model.transform(scaled_M_train)
    pca_L_train = model.transform(scaled_L_train)

    if verbose:
        print(f"Reduced H feature shape: {pca_H_train.shape}")
        print(f"Reduced M feature shape: {pca_M_train.shape}")
        print(f"Reduced L feature shape: {pca_L_train.shape}")

    end = datetime.now()
    if verbose:
        print(f"Time taken for PCA transform: {str(end-start)}")

    return pca_H_train, pca_M_train, pca_L_train, model, scaler

def run_pca_for_gmm(data, pca, scaler):
    """Returns PCA transformed data
    
    Parameters
    ----------
    data: numpy array
        The data to transform
    pca: sklearn model
        Fitted PCA model
    scaler: sklearn model
        Fitted Scaler model
    
    Returns
    -------
    transformed_data: numpy array
        PCA transformed data
    """
    scaled_data = scaler.transform(data)
    transformed_data = pca.transform(scaled_data)
    return transformed_data

def fit_gmm(h_train, m_train, l_train, h_comps, m_comps, l_comps, bayesian, cov_mat='diag', iters=1000, verbose=False, verbose_interval=100):
    """Fits and returns GMMs based on provided data

    Parameters
    ----------
    h_train: numpy array
        Training features for H
    m_train: numpy array
        Training features for M
    l_train: numpy array
        Training features for L
    h_comps: int
        Number of components for GMM H
    m_comps: int
        Number of components for GMM M
    l_comps: int
        Number of components for GMM L
    bayesian: bool
        Whether to train a bayesian model (default = False)
    cov_mat: str
        Covariance matrix type (default = diag)
    iters: int
        Number of iterations per GMM (default = 1000)
    verbose: bool
        Verbosity of the function (default = False)
    verbose_interval: bool
        Sklearn parameter verbose interval for GMM (default = 100)

    Returns
    -------
    gmm_h
        Fitted GMM model for H
    gmm_m
        Fitted GMM model for M
    gmm_l
        Fitted GMM model for L
    """
    start = datetime.now()

    if bayesian:
        gmm_h = BayesianGaussianMixture(n_components=h_comps, max_iter=iters, covariance_type=cov_mat, verbose=verbose, verbose_interval=verbose_interval, random_state=42).fit(h_train)
        gmm_m = BayesianGaussianMixture(n_components=m_comps, max_iter=iters, covariance_type=cov_mat, verbose=verbose, verbose_interval=verbose_interval, random_state=42).fit(m_train)
        gmm_l = BayesianGaussianMixture(n_components=l_comps, max_iter=iters, covariance_type=cov_mat, verbose=verbose, verbose_interval=verbose_interval, random_state=42).fit(l_train)
    else:
        gmm_h = GaussianMixture(n_components=h_comps, max_iter=iters, covariance_type=cov_mat, verbose=verbose, verbose_interval=verbose_interval, random_state=42).fit(h_train)
        gmm_m = GaussianMixture(n_components=m_comps, max_iter=iters, covariance_type=cov_mat, verbose=verbose, verbose_interval=verbose_interval, random_state=42).fit(m_train)
        gmm_l = GaussianMixture(n_components=l_comps, max_iter=iters, covariance_type=cov_mat, verbose=verbose, verbose_interval=verbose_interval, random_state=42).fit(l_train)

    end = datetime.now()
    
    if verbose:
        print(f"Time taken for fitting all GMMs: {str(end-start)}")

    return gmm_h, gmm_m, gmm_l

def get_likelihood_scores_for_samples(samples, gmm_h, gmm_m, gmm_l):
    """Returns the likelihood scors for samples based on 2H - M - L formulation
    
    Parameters
    ----------
    samples: numpy array
        Test samples in the form of a numpy array
    gmm_h: sklearn model
        Trained GMM for H landuse
    gmm_m: sklearn model
        Trained GMM for M landuse
    gmm_l: sklearn model
        Trained GMM for L landuse
    
    Returns
    -------
    H_score: list/array
        Likelihood scores for H classification
    M_score: list/array
        Likelihood scores for M classification
    L_score: list/array
        Likelihood scores for L classification
    """
    h_score = gmm_h.score_samples(samples)
    m_score = gmm_m.score_samples(samples)
    l_score = gmm_l.score_samples(samples)

    H_score = 2 * h_score - m_score - l_score
    M_score = 2 * m_score - l_score - h_score
    L_score = 2 * l_score - h_score - m_score

    return H_score, M_score, L_score

def get_landuse_threshold(data, gmm_h, gmm_m, gmm_l, true_label, ptile):
    """Get threshold based on true positives from training data for a single landuse
    
    Parameters
    ----------
    data: numpy array
        training data
    gmm_h: sklearn model
        trained GMM for H
    gmm_m: sklearn model
        trained GMM for M
    gmm_l: sklearn model
        trained GMM for L
    true_label: str
        true label for the predicted data (used to get true positives)
    ptile: float
        The percentile to use as threshold
    
    Returns
    -------
    thres: float
        the threshold value based on data
    """
    # make predictions on data (training)
    pred_df = run_gmm_on_data(data, gmm_h, gmm_m, gmm_l, true_label)
    # get true positives
    true_pos = pred_df[pred_df['Target'] == pred_df['Predicted']]
    # generate percentile threshold
    thres = np.percentile(true_pos[f'{true_label} score'], ptile)
    return thres

def get_predictions_from_scores(h_scores, m_scores, l_scores, thresholds=None):
    """Returns GMM landuse label predictions given scores and a threshold
    
    Parameters
    ----------
    h_scores: numpy array/list
        scores generated from GMM H for the data
    m_scores: numpy array/list
        scores generated from GMM M for the data
    l_scores: numpy array/list
        scores generated from GMM L for the data
    thresholds: list
        list of threshold values for H, M, L scores, 
        if none, proceeds without threshold (default = None)
    
    Returns
    -------
    predicted_labels: list
        List of predicted labels for the scores
    """
    if thresholds is not None:
        h_thres = thresholds[0]
        m_thres = thresholds[1]
        l_thres = thresholds[2]
        h_scores[h_scores < h_thres] = np.nan
        m_scores[m_scores < m_thres] = np.nan
        l_scores[l_scores < l_thres] = np.nan
    ind_lab = {'0': 'H', '1': 'M', '2': 'L'}
    predicted_labels = []
    for i in range(len(h_scores)):
        if np.isnan(h_scores[i]) and np.isnan(m_scores[i]) and np.isnan(l_scores[i]):
            predicted_labels.append('UNID')
        else:
            tmp_ind = np.nanargmax([h_scores[i], m_scores[i], l_scores[i]]).astype('str')
            predicted_labels.append(ind_lab[tmp_ind]) 
    return predicted_labels

def run_gmm_on_data(data, gmm_h, gmm_m, gmm_l, true_label, thresholds=None):
    """Generates a dataframe containing targets, predictions, and likelihood scores
    
    Parameters
    ----------
    data: numpy array
        the acoustic feature data array
    gmm_h: sklearn model
        trained H gmm model
    gmm_m: sklearn model
        trained M gmm model
    gmm_l: sklearn model
        trained L gmm model
    true_label: str
        true label for the data
    thresholds: list
        list of thresholds for H, M, and L scores respectively. 
        If not provided, no thresholds are used
    """
    df = pd.DataFrame()
    h_scores, m_scores, l_scores = get_likelihood_scores_for_samples(data, gmm_h, gmm_m, gmm_l)
    predicted_labels = get_predictions_from_scores(h_scores, m_scores, l_scores, thresholds)
    true_labels = [true_label for x in predicted_labels]
    df['Target'] = true_labels
    df['Predicted'] = predicted_labels
    df['H score'] = h_scores
    df['M score'] = m_scores
    df['L score'] = l_scores
    return df

def get_true_positives(data, gmm_h, gmm_m, gmm_l, true_label):
    """Get data for true positives from training data for a single landuse
    
    Parameters
    ----------
    data: numpy array
        training data
    gmm_h: sklearn model
        trained GMM for H
    gmm_m: sklearn model
        trained GMM for M
    gmm_l: sklearn model
        trained GMM for L
    true_label: str
        true label for the predicted data (used to get true positives)
    
    Returns
    -------
    true_pos
        true positive predictions and scores
    """
    # make predictions on data (training)
    pred_df = run_gmm_on_data(data, gmm_h, gmm_m, gmm_l, true_label)
    # get true positives
    true_pos = pred_df[pred_df['Target'] == pred_df['Predicted']]
    return true_pos

def get_cm_values(res_dict, threshold):
    """Returns confusion matrices for the given threshold predictions
    
    Parameters
    ----------
    res_dict: dict
        Data dictionary for thresholded runs
    threshold: float
        Threshold percentile to use (for no thresholding use None)
    
    Returns
    -------
    cm_sub: ndarray
        confusion matrix for H, M, L
    cm_unid: ndarray
        confusion matrix for H, M, L, UNID
    """
    df = res_dict[threshold]
    cm = confusion_matrix(df['Target'], df['Predicted'], labels=['H', 'M', 'L', 'UNID'])
    cm_sub = cm[:3, :3]
    cm_sub = normalize(cm_sub, axis=1, norm='l1')
    cm_unid = normalize(cm, axis=1, norm='l1')
    return cm_sub, cm_unid

def run_model(dataset, no_folds, habitat, feat_time, threshold, verbose=False):
    """Runs model and returns a dataframe with predictions and scores
    
    Parameters
    ----------
    dataset: pandas dataframe
        the dataset to use
    no_folds: int
        number of folds for cross validation (stratified group k fold)
    habitat: str
        habitat for which to run model
    feat_time: str
        feature time in ms
    threshold: float
        threshold percentile to use (for no thresholding use None)
    verbose: bool
        flag to make function verbose
    
    Returns
    -------
    test_df: pandas dataframe
        dataframe containing targets, predictions, and GMM likelihood scores
    """
    test_df = pd.DataFrame()
    for i in range(no_folds):
        # get models and data
        scaler, pca, gmm_h, gmm_m, gmm_l = get_gmm_fold_models(habitat, feat_time, fold=i)
        h_train, m_train, l_train = extract_features_for_fold(dataset, cv_fold_no=i, train_data=True, no_folds=no_folds, verbose=verbose)
        h_train = run_pca_for_gmm(h_train, pca, scaler)
        m_train = run_pca_for_gmm(m_train, pca, scaler)
        l_train = run_pca_for_gmm(l_train, pca, scaler)
        h_test, m_test, l_test = extract_features_for_fold(dataset, cv_fold_no=i, train_data=False, no_folds=no_folds, verbose=verbose)
        h_test = run_pca_for_gmm(h_test, pca, scaler)
        m_test = run_pca_for_gmm(m_test, pca, scaler)
        l_test = run_pca_for_gmm(l_test, pca, scaler)
        
        # get thresholds
        if threshold is not None:
            h_thres = get_landuse_threshold(h_train, gmm_h, gmm_m, gmm_l, true_label='H', ptile=threshold)
            m_thres = get_landuse_threshold(m_train, gmm_h, gmm_m, gmm_l, true_label='M', ptile=threshold)
            l_thres = get_landuse_threshold(l_train, gmm_h, gmm_m, gmm_l, true_label='L', ptile=threshold)
            thresholds = [h_thres, m_thres, l_thres]
        else:
            thresholds = None
        
        if verbose:
            print(f"Thresholds: {thresholds}")

        # run gmm on test data
        h_df = run_gmm_on_data(h_test, gmm_h, gmm_m, gmm_l, 'H', thresholds=thresholds)
        m_df = run_gmm_on_data(m_test, gmm_h, gmm_m, gmm_l, 'M', thresholds=thresholds)
        l_df = run_gmm_on_data(l_test, gmm_h, gmm_m, gmm_l, 'L', thresholds=thresholds)
        test_df = pd.concat([test_df, h_df, m_df, l_df])
    return test_df

