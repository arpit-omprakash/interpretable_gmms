import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from analysis_libs import get_predictions_from_scores
from sklearn.metrics import f1_score, precision_score, recall_score

def generate_threshold_predictions(score_df, threshold_dict, no_folds, thresholds, verbose=False):
    """Returns dictionary of thresholded prediction dataframes
    
    Parameters
    ----------
    score_df: dict
        Dictionary of thresholded scores dataframes
    threshold_dict: dict
        Dictionary of absolute threshold values per fold
    no_folds: int
        Total number of folds
    thresholds: list
        List of percentile thresholds
    verbose: bool
        Verbosity of the function
    
    Returns
    -------
    dict
        Dictionary of thresholded prediction dataframes
    """
    res_dict = {}
    for threshold in thresholds:
        if verbose:
            print(f"Running for threshold {threshold}")
        df = pd.DataFrame()
        for i in range(no_folds):
            # get thresholds
            thresholds = threshold_dict[threshold][f"fold_{i}"]

            # get results for fold
            fold_df = score_df[score_df['fold'] == i]
            h_scores = np.asarray(list(fold_df['hscores']))
            m_scores = np.asarray(list(fold_df['mscores']))
            l_scores = np.asarray(list(fold_df['lscores']))
            predicted_labels = get_predictions_from_scores(h_scores, m_scores, l_scores, thresholds)
            fold_df = fold_df.assign(predicted=predicted_labels)
            df = pd.concat([df, fold_df], ignore_index=True)
        res_dict[threshold] = df
    return res_dict
        
def generate_threshold_stats(res_dict, thresholds):
    """Returns dataframe with accuracy stats for all thresholds
    
    Parameters
    ----------
    res_dict: dict
        Dictionary of thresholded prediction dataframes
    thresholds: list
        List of percentile thresholds
    
    Returns
    -------
    pandas dataframe
        Accuracy dataframe for all thresholds
    """
    df = pd.DataFrame(columns=['model', 'F1 score', 'Precision', 'Recall', '% UNID'])
    for threshold in thresholds:
        curr_df = res_dict[threshold]
        total_len = curr_df.shape[0]
        curr_df = curr_df[curr_df['predicted'] != 'UNID']
        sub_len = curr_df.shape[0]
        fscore = f1_score(curr_df['landuse'], curr_df['predicted'], labels=['H', 'M', 'L'], average='macro', zero_division=np.nan)
        pscore = precision_score(curr_df['landuse'], curr_df['predicted'], labels=['H', 'M', 'L'], average='macro', zero_division=np.nan)
        rscore = recall_score(curr_df['landuse'], curr_df['predicted'], labels=['H', 'M', 'L'], average='macro', zero_division=np.nan)
        unid_per = (total_len - sub_len)/total_len
        if threshold is None:
            threshold = 'BASE'
        df = pd.concat([df, pd.DataFrame.from_records([{"model": f"{threshold}", "F1 score": fscore, 'Precision': pscore, 'Recall': rscore, '% UNID': unid_per}])], ignore_index=True)
    df['precision_grad'] = df['Precision'].diff() 
    df['recall_grad'] = df['Recall'].diff() 
    df['f1_grad'] = df['F1 score'].diff() 
    df['unid_grad'] = df['% UNID'].diff()
    df['p_unid'] = df['precision_grad'] - df['unid_grad']
    df['r_unid'] = df['recall_grad'] - df['unid_grad']
    df['f_unid'] = df['f1_grad'] - df['unid_grad']
    return df

    

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 18
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--feat_time', type=str, default='960',
                        help="Specify the feature time in ms (default = 960ms)")
    parser.add_argument('-i', '--input_dir', default='resources/mod_data', type=str,
                        help="The directory for pre-calculated scores and thresholds (default = resources/mod_data)")
    parser.add_argument('-o', '--output_dir', default='plots', type=str,
                        help="The directory for plot output (default = plots)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Flag to make the program verbose")
    args = parser.parse_args()

    FEAT_TIME = args.feat_time
    HABITATS = ["deciduous", "evergreen", "grassland", "scrub"]
    NO_FOLDS = 4
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    VERBOSE = args.verbose
    THRESHOLDS = [None]
    for i in range(1, 100, 3):
        THRESHOLDS.append(i)

    f, ax = plt.subplots(2, 2, figsize=(16, 16))
    i = 0
    j = 0
    for habitat in HABITATS:
        with open(f"{INPUT_DIR}/gmm_fold_thresholds_{habitat}_{FEAT_TIME}ms.pkl", "rb") as f:
            threshold_dict = pickle.load(f)
        scores_df = pd.read_pickle(f"{INPUT_DIR}/gmm_fold_scores_{habitat}_{FEAT_TIME}ms.pkl")

        res_dict = generate_threshold_predictions(scores_df, threshold_dict, NO_FOLDS, THRESHOLDS, VERBOSE)
        df = generate_threshold_stats(res_dict, THRESHOLDS)
        if VERBOSE:
            print(df)
        sign_series = np.sign(df['f1_grad'])
        idx = sign_series[sign_series == -1.0].index[0] - 1
        thres = df.iloc[idx]['model']
        if VERBOSE:
            print(thres)
        df.plot(x='model', y=['F1 score'], ax=ax[i][j], legend=False)
        ax[i][j].axvline(x=idx, color='r', label='Optimal Threshold')
        ax[i][j].set_title(f"{habitat.capitalize()} habitat \nOptimal Threshold = {thres}")
        ax[i][j].set_xlabel("Threshold")
        ax[i][j].set_ylabel("F1 score")
        j += 1
        if j > 1:
            i += 1
            j = 0
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_s1.png", bbox_inches='tight')
        

