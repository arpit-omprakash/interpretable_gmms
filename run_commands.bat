ECHO This script runs all the code in order to generate all manuscript figures
ECHO Train GMM models
python train_models.py -v -ha deciduous
python train_models.py -v -ha evergreen
python train_models.py -v -ha grassland
python train_models.py -v -ha scrub
ECHO Generate GMM predictions dataframes
python predict.py -v -ha deciduous
python predict.py -v -ha evergreen
python predict.py -v -ha grassland
python predict.py -v -ha scrub
ECHO Generate GMM fold scores
python generate_scores_thresholds.py -v -ha deciduous
python generate_scores_thresholds.py -v -ha evergreen
python generate_scores_thresholds.py -v -ha grassland
python generate_scores_thresholds.py -v -ha scrub
ECHO Generate GMM thresholded predictions
python generate_scores_thresholds.py -v -th -ha deciduous
python generate_scores_thresholds.py -v -th -ha evergreen
python generate_scores_thresholds.py -v -th -ha grassland
python generate_scores_thresholds.py -v -th -ha scrub
ECHO Generate Figure 2 from manuscript
python fig2.py -v
ECHO Generate Figure 3 from manuscript
python fig3.py -v
ECHO Generate Figure 4 from manuscript
python fig4.py -v
ECHO Generate Figure S1 from manuscript
python fig_s1.py -v
ECHO Generate Figure S2 from manuscript
python fig_s2.py -v