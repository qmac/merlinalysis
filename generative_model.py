''' Runs a simple neuroimaging model fitting workflow '''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg
from nistats.design_matrix import make_design_matrix
from sklearn.metrics import accuracy_score

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
    fir_delays = [4]
    events['modulation'] = events['modulation'].fillna(0)
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


def run_analysis(image_file, event_file, mask_file=None):
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

    # Fit and compute R squareds
    alphas = np.logspace(-1, 3, 20)
    X_train, X_test, Y_train, Y_test = partition_data(X, img_data.T)
    weights, _, _, _, _ = bootstrap_ridge(X_train, Y_train, X_test, Y_test,
                                          alphas=alphas,
                                          nboots=5,
                                          chunklen=15,
                                          nchunks=10,
                                          use_corr=False)
    
    positive_activation = np.dot([X_test.max()], weights)
    negative_activation = np.dot([X_test.min()], weights)
    predicted = []
    for sample in Y_test:
        corr_positive = np.correlate(positive_activation, sample)
        corr_negative = np.correlate(negative_activation, sample)
        predicted.append(float(corr_positive > corr_negative))

    X_test = (X_test > 0.0).astype(float)
    print('Percentage of class 1: ' + str(X_test.mean()))
    print('Test accuracy score: ' + str(accuracy_score(X_test, predicted)))


if __name__ == '__main__':
    image_file = sys.argv[1]
    event_file = sys.argv[2]
    if len(sys.argv) == 4:
        mask_file = sys.argv[3]
    else:
        mask_file = None
    run_analysis(image_file, event_file, mask_file=mask_file)
