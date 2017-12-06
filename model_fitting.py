''' Runs a simple neuroimaging model fitting workflow '''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg
from nistats.design_matrix import make_design_matrix

from ridge.ridge import ridge_corr
from ridge.utils import zscore

TR = 1.5


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
    return ridge_corr(X_train, X_test, Y_train, Y_test, alphas, use_corr=False)


def run_analysis(image_file, event_file, output_file, plot=False):
    # Load and crop
    img = check_niimg(image_file, ensure_ndim=4)
    img_data = img.get_data()
    img_data = img_data[:, :, :, 17:]
    img_data = img_data[:, :, :, 27:]
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
    print('Reshaped data matrix: ' + str(Y.shape))

    # Fit and compute R squareds
    res = compute_rsquared(X, Y)
    r_squared = res[0]
    r_squared = np.array(r_squared)
    print('R squared matrix shape: ' + str(r_squared.shape))

    # Plot some random voxel results
    if plot:
        plt.plot(r_squared[:, np.random.randint(110592, size=10)])
        plt.show()

    # Output results
    best = np.reshape(np.argmax(r_squared, axis=0), (64, 64, 27, 1))[14, 23, 10]
    best = res[1][int(best)]
    best = np.reshape(best.T, (64, 64, 27, 90))
    _, xt, _, yt = partition_data(X, Y)
    yt = np.reshape(yt.T, (64, 64, 27, 90))
    good_one = yt[14, 23, 10]
    good_one_p = best[14, 23, 10]
    resid = (good_one - good_one_p).var()
    print 1 - (resid / good_one.var())
    # plt.plot(xt[:, 0], label='speech')
    plt.plot(good_one, label='actual')
    plt.plot(good_one_p, label='pred')
    plt.legend()
    plt.show()
    r_squared = np.max(r_squared, axis=0)
    r_squared[r_squared == 1.0] = 0.0
    r_squared = np.reshape(r_squared, (64, 64, 27, 1))
    r_squared_img = Nifti1Image(r_squared, affine=img.affine)
    r_squared_img.to_filename(output_file)


if __name__ == '__main__':
    image_file = sys.argv[1]
    event_file = sys.argv[2]
    output_file = sys.argv[3]
    run_analysis(image_file, event_file, output_file, plot=False)
