import os
import re
import pickle
import librosa
import argparse
import numpy as np
import librosa.display
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors, ticker
from helpers import load_habitat_dataset
from plot_libs import create_heatmap_grid, get_grid_bounds, run_pca_for_visualization


def plot_samples(letter, titles, h_ax, m_ax, l_ax, sample_files_list, sample_rate, window_length, hop_length):
    """Plots sample spectrograms in provided figure axes
    
    Parameters
    ----------
    letter: str
        One letter shorthand for habitat
    titles: list
        List of titles for spectrograms
    h_ax: matplotlib axes object
        Axes for H land use spectrogram
    m_ax: matplotlib axes object
        Axes for M land use spectrogram
    l_ax: matplotlib axes object
        Axes for L land use spectrogram
    sample_files_list: list
        List of files in samples directory
    sample_rate: int
        Audio sample rate for spectrogram
    window_length: int
        Window length for spectrogram creation
    hop_length: int
        Hop length for spectrogram creation
    """
    regexstr = f'{letter}H'
    r = re.compile(regexstr)
    h_filename = list(filter(r.match, sample_files_list))[0]
    regexstr = f'{letter}M'
    r = re.compile(regexstr)
    m_filename = list(filter(r.match, sample_files_list))[0]
    regexstr = f'{letter}L'
    r = re.compile(regexstr)
    l_filename = list(filter(r.match, sample_files_list))[0]
    sample_files = [h_filename, m_filename, l_filename]
    sample_axs = [h_ax, m_ax, l_ax]

    for k in range(len(sample_files)):
        fname = SAMPLE_DIR + r'/' + sample_files[k]
        y, sr = librosa.load(fname, sr=sample_rate)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, fmin=125, fmax=7500, n_mels=64, win_length=window_length, hop_length=hop_length)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect_db, x_axis='time', y_axis='mel', sr=sr, cmap='jet', ax=sample_axs[k], hop_length=hop_length)
        sample_axs[k].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x/1000)))
        sample_axs[k].set_xlabel("Time (sec)")
        sample_axs[k].set_ylabel("Frequency (Hz)")
        sample_axs[k].set_title(f"{titles[k]}")
        sample_axs[k].axvline(x=0.96, linestyle='--', color='black')
        sample_axs[k].axvline(x=1.92, linestyle='--', color='black')

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
    SAMPLE_DIR = 'resources/labelling/examples'
    SAMPLE_FILES_LIST = os.listdir(SAMPLE_DIR)
    SAMPLE_RATE = 16000
    WINDOW_LENGTH = int(round(SAMPLE_RATE * 0.025))
    HOP_LENGTH = int(round(SAMPLE_RATE * 0.01))

    f = plt.figure(constrained_layout=True, figsize=(20, 32))
    gs = f.add_gridspec(24, 9)
    f_d_full = f.add_subplot(gs[0:6, 0:6], adjustable='box', aspect=1)
    f_dh = f.add_subplot(gs[0:2, 6:])
    f_dm = f.add_subplot(gs[2:4, 6:])
    f_dl = f.add_subplot(gs[4:6, 6:])
    f_e_full = f.add_subplot(gs[6:12, 0:6], adjustable='box', aspect=1)
    f_eh = f.add_subplot(gs[6:8, 6:])
    f_em = f.add_subplot(gs[8:10, 6:])
    f_el = f.add_subplot(gs[10:12, 6:])
    f_g_full = f.add_subplot(gs[12:18, 0:6], adjustable='box', aspect=1)
    f_gh = f.add_subplot(gs[12:14, 6:])
    f_gm = f.add_subplot(gs[14:16, 6:])
    f_gl = f.add_subplot(gs[16:18, 6:])
    f_s_full = f.add_subplot(gs[18:, 0:6], adjustable='box', aspect=1)
    f_sh = f.add_subplot(gs[18:20, 6:])
    f_sm = f.add_subplot(gs[20:22, 6:])
    f_sl = f.add_subplot(gs[22:, 6:])
    main_axs = [f_d_full, f_e_full, f_g_full, f_s_full]
    i = 0
    for habitat in LIST_OF_HABITATS:
        if VERBOSE:
            print(f"Loading dataset for {habitat} habitat and {FEAT_TIME}ms features")
        dataset = load_habitat_dataset(habitat = habitat, feat_time = FEAT_TIME, mod_label='full_predicted_pca_0.95', data_dir='resources/mod_data')

        if VERBOSE:
            print("Loading models")
        with open(f'{INPUT_DIR}/gmm_models_{habitat}_{FEAT_TIME}ms_pca_0.95_full.pkl', 'rb') as fd:
            model_dict = pickle.load(fd)
        
        pca = model_dict['PCA']
        scaler = model_dict['Scaler']
        gmm_h = model_dict['GMM H']
        gmm_m = model_dict['GMM M']
        gmm_l = model_dict['GMM L']

        gmms = [gmm_h, gmm_m, gmm_l]
        # removing unid observations
        dataset = dataset[dataset['thres_pred'] != 'UNID']

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
        alphas = [0.5, 0.4, 0.3]

        for val, gmm_mean, gmm, col, cl, label, alpha in zip(grid_values, means_red, gmms, color_list, c_list, labels, alphas):
            im = main_axs[i].imshow(val, extent=[x_min, x_max, y_min, y_max], cmap=col, norm=colors.SymLogNorm(linthresh=1, linscale=0.05, vmin=np.nanmin(val), vmax=np.nanmax(val), base=10), alpha=alpha, label=label, aspect='auto')
        if habitat == 'deciduous':
            h_comp = 82 
            m_comp = 65 
            l_comp = 98 
            letter = 'D'
            titles = ['Insect calls', 'Silence', 'Agricultural sprinkler']
            plot_samples(letter, titles, f_dh, f_dm, f_dl, SAMPLE_FILES_LIST, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
        if habitat == 'grassland':
            h_comp = 53 
            m_comp = 72 
            l_comp = 82 
            letter = 'G'
            titles = ['Low-frequency insect calls', 'High-frequency insect calls', 'Wind noise, Low-frequency birds']
            plot_samples(letter, titles, f_gh, f_gm, f_gl, SAMPLE_FILES_LIST, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
        if habitat == 'evergreen':
            h_comp = 76 
            m_comp = 71 
            l_comp = 10 
            letter = 'E'
            titles = ['Low-frequency insect calls', 'Artefact from bat calls', 'Broad-range insect calls']
            plot_samples(letter, titles, f_eh, f_em, f_el, SAMPLE_FILES_LIST, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)
        if habitat == 'scrub':
            h_comp = 88 
            m_comp = 48 
            l_comp = 6 
            letter = 'T'
            titles = ['Rustling leaves', 'Wind noise, Knocking noises', 'High-frequency insect calls']
            plot_samples(letter, titles, f_sh, f_sm, f_sl, SAMPLE_FILES_LIST, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH)

        # plot the main figure components
        main_axs[i].scatter(means_red[0][h_comp, 0], means_red[0][h_comp, 1], marker="X", s=250, c='Darkblue')
        main_axs[i].scatter(means_red[1][m_comp, 0], means_red[1][m_comp, 1], marker="X", s=250, c='Brown')
        main_axs[i].scatter(means_red[2][l_comp, 0], means_red[2][l_comp, 1], marker="X", s=250, c='Orange')
        main_axs[i].set_ylabel("PC 2")
        main_axs[i].set_xlabel("PC 1")
        main_axs[i].set_title(f"{habitat.capitalize()} habitat")    
        i += 1

    if VERBOSE:
        print("Saving plot to output directory")
    plt.savefig(f"{OUTPUT_DIR}/fig4.png")