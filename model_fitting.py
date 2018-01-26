''' Runs a simple neuroimaging model fitting workflow '''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg
from nistats.design_matrix import make_design_matrix

from ridge.ridge import bootstrap_ridge
from ridge.utils import zscore
from detrend_sgolay import sgolay_filter_volume

TR = 1.5


def detrend_data(Y):
    return sgolay_filter_volume(Y, filtlen=181, degree=3)


def get_design_matrix(event_file, n_scans):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    print('Raw events shape: ' + str(events.shape))
    start_time = 0.0
    end_time = (n_scans - 1) * TR
    frame_times = np.linspace(start_time, end_time, n_scans)
    fir_delays = [1, 2, 3, 4]
    dm = make_design_matrix(frame_times, events, hrf_model='fir',
                            fir_delays=fir_delays, drift_model=None)
    dm = dm.drop('constant', axis=1)
    return dm


def partition_data(X, Y):
    test_ranges = [(185, 215), (485, 515), (785, 815)]
    delete_indices = []
    test_indices = []
    for r in test_ranges:
        delete_indices += range(r[0] - 5, r[1] + 5)
        test_indices += range(r[0], r[1])
    X_train = np.delete(X, delete_indices, axis=0)
    X_test = X[test_indices]
    Y_train = np.delete(Y, delete_indices, axis=0)
    Y_test = Y[test_indices]
    return X_train, X_test, Y_train, Y_test


def compute_rsquared(X, Y):
    alphas = np.logspace(-1, 3, 20)
    X_train, X_test, Y_train, Y_test = partition_data(X, Y)
    wt, corrs, _, _, _ = bootstrap_ridge(X_train, Y_train, X_test, Y_test,
                                         alphas=alphas,
                                         nboots=5,
                                         chunklen=15,
                                         nchunks=10,
                                         use_corr=False)
    return wt, corrs


def run_analysis(image_file, event_file, output_file, plot=False):
    # Load and crop
    img = check_niimg(image_file, ensure_ndim=4)
    img_data = img.get_data()
    img_data = img_data[:, :, :, 17:]  # Initial scanner setup
    img_data = img_data[:, :, :, 27:]  # Starting cartoon
    img_data = img_data[:, :, :, :975]
    print('Data matrix shape: ' + str(img_data.shape))

    # Get design matrix from nistats
    dm = get_design_matrix(event_file, img_data.shape[3])
    print('Design matrix shape: ' + str(dm.shape))

    # Normalize data and design matrix
    X = zscore(dm.as_matrix().T).T
    if plot:
        plt.plot(X)
        plt.show()
    Y = np.reshape(img_data, (110592, img_data.shape[3]))
    Y = zscore(Y).T
    Y = detrend_data(Y)
    print('Reshaped data matrix: ' + str(Y.shape))

    # Fit and compute R squareds
    weights, r_squared = compute_rsquared(X, Y)
    print('R squared matrix shape: ' + str(r_squared.shape))

    # Output results
    r_squared[r_squared == 1.0] = 0.0
    r_squared = np.reshape(r_squared, (64, 64, 27, 1))
    r_squared_img = Nifti1Image(r_squared, affine=img.affine)
    r_squared_img.to_filename(output_file)

    if plot:
        _, xt, _, yt = partition_data(X, Y)
        aud_vox_xyz = (14, 23, 10)
        aud_vox_idx = np.unravel_index(np.ravel_multi_index(aud_vox_xyz, (64, 64, 27)), (110592))
        voxel_actual = yt.T[aud_vox_idx]
        voxel_pred = np.dot(xt, weights).T[aud_vox_idx]
        plt.plot(xt[:, 0], label='speech')
        plt.plot(voxel_actual, label='actual')
        plt.plot(voxel_pred, label='pred')
        print(list(dm.columns))
        print('Weights for auditory cortex voxel: ' + str(weights.T[aud_vox_idx]))
        print(r_squared[aud_vox_xyz])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    image_file = sys.argv[1]
    event_file = sys.argv[2]
    output_file = sys.argv[3]
    run_analysis(image_file, event_file, output_file, plot=False)
