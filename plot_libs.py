import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

def get_grid_bounds(X_train_red):
    """Returns x and y min and max values for grid boundaries
    
    Parameters
    ----------
    X_train_red: numpy array
        The reduced features for which to extract boundaries

    Returns
    -------
    x_min: float
        lower bound of x
    x_max: float
        upper bound of x
    y_min: float
        lower bound of y
    y_max: float
        upper bound of y
    """
    x_min = np.min(X_train_red[:,0]) - 0.1
    x_max = np.max(X_train_red[:,0]) + 0.1
    y_min = np.min(X_train_red[:,1]) - 0.1
    y_max = np.max(X_train_red[:,1]) + 0.1

    return x_min, x_max, y_min, y_max

def create_heatmap_grid(dataset, X_train_red, resolution):
    """Creates a grid for heatmap visualization from PCA reduced data
    
    Parameters
    ----------
    dataset: pandas dataframe
        The dataset with reduced features
    X_train_red: numpy array
        PCA reduced features
    resolution: int
        Resolution of the grid, higher resolution produces more points on the grid
    
    Returns
    -------
    rot_vals: list
        List of three matrices containing grid values for H, M and L landuse
    """
    x_min, x_max, y_min, y_max = get_grid_bounds(X_train_red)

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)

    H = np.empty(shape=(resolution-1, resolution-1))
    H[:] = np.nan
    M = np.empty(shape=(resolution-1, resolution-1))
    M[:] = np.nan
    L = np.empty(shape=(resolution-1, resolution-1))
    L[:] = np.nan

    for i in range(len(x)-1):
        for j in range(len(y)-1):
            curr_view = dataset[(dataset['Comp 1'] < x[i+1]) & (dataset['Comp 1'] > x[i]) & (dataset['Comp 2'] < y[j+1]) & (dataset['Comp 2'] > y[j])]
            H[i][j] = np.mean(curr_view['H score'])
            M[i][j] = np.mean(curr_view['M score'])
            L[i][j] = np.mean(curr_view['L score'])

    vals = [H, M, L]
    rot_vals = []
    # rotate the matrices to align with coordinates
    # anticlockwise 90 deg
    for matrix in vals:
        rot1 = np.asarray(list(zip(*matrix))[::-1])
        rot_vals.append(rot1)

    return rot_vals

def run_pca_for_visualization(features, reduced_dims=2, model=None, fit=False):
    """Fits a PCA model or returns reduced features for a given model for visualization
    
    Parameters
    ----------
    features: numpy array
        The high dimensional features matrix
    reduced_dims: int
        Number of dimensions for visualization (default = 2)
    model: sklearn model
        Already trained PCA model for transformation
    fit: bool
        If true, fits the model first before transforming features
    
    Returns
    -------
    reduced_feats: numpy array
        features in reduced dimensions
    model: sklearn model
        Fitted PCA model
    """
    if model is None:
        model = PCA(n_components = reduced_dims)
    if fit:
        reduced_feats = model.fit_transform(features)
    else:
        reduced_feats = model.transform(features)
    return reduced_feats, model

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    '''
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    '''
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip, ax