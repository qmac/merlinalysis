''' Runs a simple neuroimaging model fitting workflow '''

import sys
import numpy as np
import pandas as pd

from nilearn._utils.niimg_conversions import check_niimg
from nistats.design_matrix import make_design_matrix

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

TR = 1.5


def get_labels(event_file, n_scans):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    events = events.sort_values('onset').reset_index(drop=True)
    events = events[events['trial_type'] == 'person.n.01']
    start_time = 0.0
    end_time = (n_scans - 1) * TR
    frame_times = np.linspace(start_time, end_time, n_scans)
    dm = make_design_matrix(frame_times, events, hrf_model='fir', drift_model=None, fir_delays=[0])
    dm = dm.drop('constant', axis=1)
    return dm


def get_data_matrix(data):
    fir_delays = [3, 4, 5]
    rolls = []
    for f in fir_delays:
        roll = np.roll(data.T, -f, axis=0)
        roll[-f:] = 0
        rolls.append(roll)
    return np.hstack(rolls)


def partition_data(X, Y):
    test_ranges = [(185, 215), (485, 515), (785, 815)]
    delete_indices = []
    test_indices = []
    for r in test_ranges:
        delete_indices += range(r[0] - 6, r[1] + 6)
        test_indices += range(r[0], r[1])
    X_train = np.delete(X, delete_indices, axis=0)
    X_test = X[test_indices]
    Y_train = np.delete(Y, delete_indices, axis=0)
    Y_test = Y[test_indices]
    return X_train, X_test, Y_train, Y_test


def run_single_subject(image_file, event_file, mask_file):
    # Load imaging data
    mask = check_niimg(mask_file, ensure_ndim=3).get_data().astype(bool)
    print('Mask shape: ' + str(mask.shape))
    print('Number of masked voxels: ' + str(np.sum(mask)))

    labels = np.around(get_labels(event_file, 975).as_matrix()).T[0]

    img = check_niimg(image_file, ensure_ndim=4)
    img_data = img.get_data()[mask]
    rolled = get_data_matrix(img_data)
    X_train, X_test, Y_train, Y_test = partition_data(rolled, labels)
    print('Training matrix: ' + str(X_train.shape))
    print('Testing matrix: ' + str(X_test.shape))
    print('Training labels shape: ' + str(Y_train.shape))
    print('Testing labels shape: ' + str(Y_test.shape))

    # Reduce dimensionality
    dim_red = SelectKBest(f_classif, k=1000)
    X_train = dim_red.fit_transform(X_train, Y_train)
    X_test = dim_red.transform(X_test)
    print('Dimensionality reduced, new data shape: ' + str(X_train.shape))

    # Run classification
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    y_preds = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:,1]
    test_acc_score = accuracy_score(Y_test, y_preds)
    auc_score = roc_auc_score(Y_test, y_probs)
    print('Test accuracy score: ' + str(test_acc_score))
    print('AUC score: ' + str(auc_score))
    print('Percentage of class 1: ' + str(Y_test.mean()))

def run_analysis(image_files, event_file, mask_file):
    # Load imaging data
    mask = check_niimg(mask_file, ensure_ndim=3).get_data().astype(bool)
    print('Mask shape: ' + str(mask.shape))
    print('Number of masked voxels: ' + str(np.sum(mask)))

    X_train = []
    for image_file in image_files[1:11]:
        img = check_niimg(image_file, ensure_ndim=4)
        img_data = img.get_data()[mask]
        rolled = get_data_matrix(img_data)
        X_train.append(rolled)
        print('Done loading image file: ' + str(image_file))
    X_train = np.concatenate(X_train, axis=0)
    print('Training matrix: ' + str(X_train.shape))

    X_test = check_niimg(image_files[0], ensure_ndim=4).get_data()[mask]
    X_test = get_data_matrix(X_test)
    print('Testing matrix: ' + str(X_test.shape))

    # Get labels
    labels = np.around(get_labels(event_file, 975).as_matrix()).T[0]
    Y_train = np.concatenate([labels] * 10)#(len(image_files) - 1))
    print('Labels shape: ' + str(Y_train.shape))
    print('Percentage of class 1: ' + str(labels.mean()))

    # Reduce dimensionality
    dim_red = SelectKBest(f_classif, k=1000)
    X_train = dim_red.fit_transform(X_train, Y_train)
    X_test = dim_red.transform(X_test)
    print('Dimensionality reduced, new data shape: ' + str(X_train.shape))

    # Re-run classification
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    y_preds = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:,1]
    test_acc_score = accuracy_score(labels, y_preds)
    auc_score = roc_auc_score(labels, y_probs)
    print('Test accuracy score: ' + str(test_acc_score))
    print('AUC score: ' + str(auc_score))


if __name__ == '__main__':
    event_file = sys.argv[1]
    mask_file = sys.argv[2]
    image_files = sys.argv[3:]
    # run_analysis(image_files, event_file, mask_file)
    run_single_subject(image_files[0], event_file, mask_file)
