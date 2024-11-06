import pickle
import argparse
import numpy as np
import pandas as pd
import natsort as ns
import matplotlib as mpl
from natsort import index_natsorted
from analysis_libs import run_model
from matplotlib import pyplot as plt
from helpers import load_habitat_dataset
from sklearn.metrics import f1_score, precision_score, recall_score

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 20
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--feat_time', type=str, default='960',
                        help="Specify the feature time in ms (default = 960ms)")
    parser.add_argument('-o', '--output_dir', default='plots', type=str,
                        help="The directory for plot output (default = plots)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Flag to make the program verbose")
    args = parser.parse_args()

    FEAT_TIME = args.feat_time
    NO_FOLDS = 4
    OUTPUT_DIR = args.output_dir
    VERBOSE = args.verbose
    LIST_OF_HABITATS = ['deciduous', 'evergreen', 'grassland', 'scrub']

    f, axs = plt.subplots(2, 2, figsize=(16, 16))
    i = 0
    j = 0
    for habitat in LIST_OF_HABITATS:
        THRESHOLDS = [None, 10, 20, 30, 40, 50, 60, 70]
        if habitat == 'grassland':
            THRESHOLDS.append(76)
            OP_THRES = 76
        elif habitat == 'deciduous':
            THRESHOLDS.append(64)
            OP_THRES = 64
        elif habitat == 'scrub':
            THRESHOLDS.append(22)
            OP_THRES = 22
        elif habitat == 'evergreen':
            THRESHOLDS.append(43)
            OP_THRES = 43
        if VERBOSE:
            print(f"Loading dataset for {habitat} habitat and {FEAT_TIME}ms features")
        dataset = load_habitat_dataset(habitat=habitat, feat_time=FEAT_TIME)
        try:
            if VERBOSE:
                print()
                print("Loading thresholded predictions")
            with open(f'resources/mod_data/{habitat}_{FEAT_TIME}ms_thresholded_model_runs.pkl', 'rb') as fd:
                res_dict = pickle.load(fd)
        except FileNotFoundError:
            print("No saved file found!")
            print("Running models")
            res_dict = {}
            for threshold in THRESHOLDS:
                if VERBOSE:
                    print()
                    print(f"Running for threshold {threshold}")
                test_df = run_model(dataset, NO_FOLDS, habitat, FEAT_TIME, threshold, VERBOSE)
                res_dict[threshold] = test_df
            if VERBOSE:
                print("Saving model outputs")
            with open(f'resources/mod_data/{habitat}_{FEAT_TIME}ms_thresholded_model_runs.pkl', 'wb') as fd:
                pickle.dump(res_dict, fd)
       
        df = pd.DataFrame(columns=['model', 'F1 score', 'Precision', 'Recall', '% UNID'])
        for threshold in THRESHOLDS:
            if threshold != OP_THRES:
                curr_df = res_dict[threshold]
                total_len = curr_df.shape[0]
                curr_df = curr_df[curr_df['Predicted'] != 'UNID']
                sub_len = curr_df.shape[0]
                fscore = f1_score(curr_df['Target'], curr_df['Predicted'], labels=['H', 'M', 'L'], average='macro', zero_division=np.nan)
                pscore = precision_score(curr_df['Target'], curr_df['Predicted'], labels=['H', 'M', 'L'], average='macro', zero_division=np.nan)
                rscore = recall_score(curr_df['Target'], curr_df['Predicted'], labels=['H', 'M', 'L'], average='macro', zero_division=np.nan)
                unid_per = (total_len - sub_len)/total_len
                if threshold is None:
                    threshold = 'BASE'
                df = pd.concat([df, pd.DataFrame.from_records([{"model": f"{threshold}", "F1 score": fscore, 'Precision': pscore, 'Recall': rscore, '% UNID': unid_per}])], ignore_index=True)
                
        if VERBOSE:
            print(df)
        df.sort_values(by=['model'], key=lambda x: np.argsort(index_natsorted(df['model'], alg=ns.NA)), inplace=True)
        df.plot(kind='bar', x='model', ax = axs[i][j], rot=50)
        axs[i][j].set_title(f"{habitat.capitalize()} habitat")
        axs[i][j].set_xlabel("")
        axs[i][j].set_ylabel("Proportion")
        axs[i][j].legend()
        j += 1
        if j > 1:
            i += 1
            j = 0
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_s2.png", bbox_inches='tight')

