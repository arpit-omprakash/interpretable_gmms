import pickle
import argparse
import numpy as np
import pandas as pd
import natsort as ns
import seaborn as sns
import matplotlib as mpl
from natsort import index_natsorted
from matplotlib import pyplot as plt
from helpers import load_habitat_dataset
from analysis_libs import get_cm_values, run_model
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

    f, axs = plt.subplots(3, 4, figsize=(26, 16))
    for habitat in LIST_OF_HABITATS:
        THRESHOLDS = [None, 10, 20, 30, 40, 50, 60, 70]
        if habitat == 'grassland':
            THRESHOLDS.append(76)
            OP_THRES = 76
            j = 0
        elif habitat == 'deciduous':
            THRESHOLDS.append(64)
            OP_THRES = 64
            j = 1
        elif habitat == 'scrub':
            THRESHOLDS.append(22)
            OP_THRES = 22
            j = 2
        elif habitat == 'evergreen':
            THRESHOLDS.append(43)
            OP_THRES = 43
            j = 3
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
        # first line without threshold cm
        cm_sub, cm_unid = get_cm_values(res_dict, None)
        sns.heatmap(cm_sub, annot=True, xticklabels=['Ref', 'Dist', 'Agri'], yticklabels=['Ref', 'Dist', 'Agri'], ax=axs[0][j], cmap='Blues', vmin=0, vmax=1)
        axs[0][j].set_title(f'{habitat.capitalize()} habitat')
        
        # second line fpr plot
        df = pd.DataFrame(columns=['model', 'Precision', '% UNID'])
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
                df = pd.concat([df, pd.DataFrame.from_records([{"model": f"{threshold}", 'Precision': pscore, '% UNID': unid_per}])], ignore_index=True)
            
        if VERBOSE:
            print(df)
        df.sort_values(by=['model'], key=lambda x: np.argsort(index_natsorted(df['model'], alg=ns.NA)), inplace=True)
        df.plot(kind='bar', x='model', ax = axs[1][j], rot=50)
        axs[1][j].set_xlabel("")
        axs[1][j].set_ylabel("Proportion")
        axs[1][j].legend()

        # third line thresholded cm
        cm_sub_thres, cm_unid = get_cm_values(res_dict, OP_THRES)
        cm_plot = cm_sub_thres - cm_sub
        cm_plot = np.concatenate((cm_plot, cm_unid[:,3][:3].reshape(3, 1)), axis=1)
        mask = np.zeros((3, 4))
        mask[:,3] = True
        sns.heatmap(cm_plot, annot=True, ax=axs[2][j], cbar=False, cmap='Greys', vmin=0, vmax=1)
        sns.heatmap(cm_plot, mask=mask, ax=axs[2][j], cmap='bwr_r', xticklabels=['Ref', 'Dist', 'Agri', 'UNID'], yticklabels=['Ref', 'Dist', 'Agri'], annot_kws={'color': 'white'}, color='white')
        
    f.text(0.0, 0.8, 'CM without \nthresholding', verticalalignment='center', rotation=90, fontsize=20)
    axs[1][0].set_title('Model accuracy at \nvarious thresholds', rotation='vertical',x=-0.3,y=0.1)
    axs[2][0].set_title('Change in values at \noptimal threshold', rotation='vertical',x=-0.3,y=0.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3.png", bbox_inches='tight')

