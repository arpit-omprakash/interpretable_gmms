import pickle
import argparse
import numpy as np
import matplotlib as mpl
from helpers import load_habitat_dataset
from matplotlib import colors, pyplot as plt
from plot_libs import create_heatmap_grid, get_grid_bounds, run_pca_for_visualization

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 18
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', '--resolution', type=int, default=50,
                        help="Resolution for creating the heatmap grids (default = 50)")
    parser.add_argument('-o', '--output_dir', type=str, default=r'plots',
                        help="Path to save output images to (default = plots)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Flag to make the program verbose")
    args = parser.parse_args()

    FEAT_TIME = '960'
    VERBOSE = args.verbose
    RESOLUTION = args.resolution
    INPUT_DIR = 'resources/models'
    OUTPUT_DIR = args.output_dir
    LIST_OF_HABITATS = ['deciduous', 'evergreen', 'grassland', 'scrub']

    f, axs = plt.subplots(2, 2, figsize=(16, 12))
    i = 0
    j = 0
    for habitat in LIST_OF_HABITATS:
        if VERBOSE:
            print(f"Loading dataset for {habitat} habitat and {FEAT_TIME}ms features")
        dataset = load_habitat_dataset(habitat = habitat, feat_time = FEAT_TIME, mod_label='full_predicted_pca_0.95', data_dir='resources/mod_data')

        if VERBOSE:
            print("Loading models")
        with open(f'{INPUT_DIR}/gmm_models_{habitat}_{FEAT_TIME}ms_pca_0.95_full.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        
        pca = model_dict['PCA']
        scaler = model_dict['Scaler']
        gmm_h = model_dict['GMM H']
        gmm_m = model_dict['GMM M']
        gmm_l = model_dict['GMM L']

        gmms = [gmm_h, gmm_m, gmm_l]

        X_train_pca = np.asarray(list(dataset['pca_feats']))
        X_train_red, pca_viz = run_pca_for_visualization(X_train_pca, fit=True)
        means_red = [pca_viz.transform(gmm_h.means_), pca_viz.transform(gmm_m.means_), pca_viz.transform(gmm_l.means_)]
        dataset['Comp 1'] = X_train_red[:,0]
        dataset['Comp 2'] = X_train_red[:,1]

        grid_values = create_heatmap_grid(dataset, X_train_red, RESOLUTION)

        x_min, x_max, y_min, y_max = get_grid_bounds(X_train_red)
    
        color_list = [colors.LinearSegmentedColormap.from_list("", ["white","blue"]), 
                    colors.LinearSegmentedColormap.from_list("", ["white","red"]), 
                    colors.LinearSegmentedColormap.from_list("", ["white","yellow"])]
        c_list = ['Darkblue', 'Brown', 'Orange']
        labels = ["Reference", "Disturbed", "Agriculture"]
        alphas = [0.5, 0.4, 0.4]

        for val, gmm_mean, gmm, col, cl, label, alpha in zip(grid_values, means_red, gmms, color_list, c_list, labels, alphas):
            im = axs[i][j].imshow(val, extent=[x_min, x_max, y_min, y_max], cmap=col, norm=colors.SymLogNorm(linthresh=1, linscale=0.05, vmin=np.nanmin(val), vmax=np.nanmax(val), base=10), alpha=alpha, label=label, aspect='auto')
        axs[i][j].set_ylabel("PC 2")
        axs[i][j].set_xlabel("PC 1")
        axs[i][j].set_title(f"{habitat.capitalize()} habitat")
        i += 1
        if i > 1:
            i = 0
            j += 1
        
    if VERBOSE:
        print("Saving heatmaps to output directory")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2.png", bbox_inches="tight")