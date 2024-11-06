# Interpretable GMMs

### Setup

Code is tested using Python 3.10.9 on an Anaconda installation in Windows 11.  
Run the following code to create and activate a new conda environment with all the required packages:

```
$ conda create --name <envname> --file requirements.txt
$ conda activate <envname>
```

The audio feature datasets containing VGGish features for audio from various habitats can be downloaded from here: https://doi.org/10.5281/zenodo.13772138 

To calculate VGGish features for your own audio files you may refer to the repository below:  
Sarab Sethi. (2020). sarabsethi/audioset_soundscape_feats_sethi2019: June 2020 release (1.2). Zenodo. https://doi.org/10.5281/zenodo.3907296

### Usage

All python scripts are provided with a help message containing usage instructions that can be accessed by using the `-h` flag following the file name. For example, the following code shows the usage description for the file `train_models.py`:

```
$ python train_models.py -h
usage: train_models.py [-h] [-ha HABITAT] [-o OUTPUT_DIR] [-v]

options:
  -h, --help            show this help message and exit
  -ha HABITAT, --habitat HABITAT
                        Specify the habitat (default = grassland)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to save models (default = resources/models)
  -v, --verbose         Flag to make the program verbose
```

The python scripts to reproduce figures/analyses from our paper are:

* `train_models.py` trains the GMM models using audio features for a given habitat. The trained models are stored in the `resources/models` directory by default. Make sure this directory is created before running the script. This script is run first to train GMM models for any further scripts.

* `predict.py` generates dataframes with GMM predictions and scores of both baseline and optimally thresholded models for a given habitat. The dataframes are stored in `resources/mod_data` directory by default. Make sure this directory is created before running the script. This script is run second to make predictions using the trained GMMs.

* `generate_scores_thresholds.py` generates scores and absolute threshold values for percentile thresholds ranging from 1 to 100 in steps of 3. The generated dataframes are stored in `resources/mod_data` directory by default. Make sure this directory is created before running the script. This script needs to be run twice to generate scores and thresholds. Refer to help section of the script for more information.

* `fig3.py` creates 2D heatmaps of audio features coloured by land use across habitats for the baseline model. PCA is used as a dimensionality reduction technique.

* `fig4.py` creates three panels of a larger figure. (1) confusion matrices for the baseline GMM model across habitats. (2) model precision at various percentile thresholds across habitats. (3) change in confusion matrices for the optimally thresholded model vs baseline.

* `fig5.py` creates 2D heatmaps of audio features coloured by land use across habitats for the optimally thresholded model using PCA. Also plots example spectrograms from each land use for all habitats besides the corresponding heatmaps. The spectrogram locations are highlighted as crosses on the 2D heatmaps.

* `fig_s1.py` plots F1 score vs threshold for all habitats. The optimal thresholds are highlighted by red vertical lines in each subplot.

* `fig_s2.py` plots model precision, recall, F1 score, percentage UNID predictions at various percentile thresholds across habitats.

* `helpers.py`, `analysis_libs.py`, and `plot_libs.py` contain auxiliary functions


To simplify the process, a batch file called `run_commands.bat` is provided that runs all the scripts in appropriate order to create figures from the manuscript. You can run this using:

```
$ run_commands.bat
```
