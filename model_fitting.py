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

TR = 1.5


def get_design_matrix(event_file, n_scans):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    events = events.sort_values('onset').reset_index(drop=True)
    print('Raw events shape: ' + str(events.shape))
    start_time = 0.0
    end_time = (n_scans - 1) * TR
    frame_times = np.linspace(start_time, end_time, n_scans)
    fir_delays = [1, 2, 3, 4, 5]
    events['modulation'] = events['modulation'].fillna(0)
    dm = make_design_matrix(frame_times, events, hrf_model='fir',
                            fir_delays=fir_delays, drift_model=None)
    dm = dm.drop('constant', axis=1)
    return dm


def partition_data(X, Y, speech_only=False):
    if speech_only:
        speech_events = get_design_matrix('events/audio_speech_events.csv', 975)
        speech_indices = (speech_events != 0).any(axis=1)
        X = X[speech_indices.values]
        Y = Y[speech_indices.values]
        test_ranges = [(105, 135), (330, 360), (555, 585)]
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
    else:
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
    corrs = np.sign(corrs) * np.power(corrs, 2)
    return wt, corrs


def run_analysis(image_file, event_file, output_file, mask_file=None, plot=False):
    # Load and crop
    img = check_niimg(image_file, ensure_ndim=4)
    img_data = img.get_data()
    print('Data matrix shape: ' + str(img_data.shape))

    if mask_file:
        mask = check_niimg(mask_file, ensure_ndim=3).get_data().astype(bool)
        img_data = img_data[mask]
        print('Masked data matrix shape: ' + str(img_data.shape))
    else:
        img_data = np.reshape(img_data, (245245, img_data.shape[3]))

    # Get design matrix from nistats
    dm = get_design_matrix(event_file, img_data.shape[1])
    print('Design matrix shape: ' + str(dm.shape))

    # Normalize data and design matrix
    X = zscore(dm.as_matrix().T).T
    if plot:
        plt.plot(X)
        plt.show()

    # Fit and compute R squareds
    weights, r_squared = compute_rsquared(X, img_data.T)
    print('R squared matrix shape: ' + str(r_squared.shape))

    # Output results
    if mask_file:
        output = np.zeros((65, 77, 49))
        output[mask] = r_squared
    else:
        output = np.reshape(r_squared, (65, 77, 49))
        output[output == 1.0] = 0.0
    r_squared_img = Nifti1Image(output, affine=img.affine)
    r_squared_img.to_filename(output_file)


if __name__ == '__main__':
    image_file = sys.argv[1]
    event_file = sys.argv[2]
    output_file = sys.argv[3]
    if len(sys.argv) == 5:
        mask_file = sys.argv[4]
    else:
        mask_file = None
    run_analysis(image_file, event_file, output_file, mask_file=mask_file, plot=False)
