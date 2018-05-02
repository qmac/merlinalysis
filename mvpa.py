''' Runs a simple neuroimaging model fitting workflow '''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn._utils.niimg_conversions import check_niimg
from nistats.design_matrix import make_design_matrix

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.svm import LinearSVC

TR = 1.5


def get_multiclass_labels(event_file):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    events = events.sort_values('onset').reset_index(drop=True)
    dm = events.drop(['onset', 'duration'], axis=1)
    return dm


def get_labels(event_file, n_scans, trial_type=None):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    events = events.sort_values('onset').reset_index(drop=True)
    if trial_type:
        events = events[events['trial_type'] == trial_type]
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


def partition_indices():
    test_ranges = [(185, 215), (485, 515), (785, 815)]
    delete_indices = []
    test_indices = []
    for r in test_ranges:
        delete_indices += range(r[0] - 6, r[1] + 6)
        test_indices += range(r[0], r[1])
    train_indices = np.delete(range(975), delete_indices, axis=0)
    return train_indices, test_indices


def partition_data(X, Y):
    train_indices, test_indices = partition_indices()
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]


def run_single_subject(image_file, event_file, mask_file, trial_type=None,
                       multiclass=False, permutation_test=False, plot=False):
    # Load imaging data
    mask = check_niimg(mask_file, ensure_ndim=3).get_data().astype(bool)
    print('Mask shape: ' + str(mask.shape))
    print('Number of masked voxels: ' + str(np.sum(mask)))

    if multiclass:
        labels = get_multiclass_labels(event_file).as_matrix().T[0]
    else:
        labels = np.around(get_labels(event_file, 975, trial_type=trial_type).as_matrix()).T[0]
        if (labels[partition_indices()[1]].sum()) < 3:
            raise ValueError('This label does not occur frequently enough')

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
    test_acc_score = accuracy_score(Y_test, y_preds)
    print('Test accuracy score: ' + str(test_acc_score))
    print('Percentage of class 1: ' + str(max(np.unique(Y_test, return_counts=True)[1]) / float(len(Y_test))))

    if not multiclass:
        y_probs = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(Y_test, y_probs)
        auc_score = auc(fpr, tpr)
        print('AUC score: ' + str(auc_score))
        if plot:
            if trial_type == 'art.n.01':
                plt.plot(fpr, tpr, color='darkorange')
                with open('art_result.txt', 'w') as f:
                    f.write(str(Y_test) + '\n')
                    f.write(str(y_probs) + '\n')
            else:
                plt.plot(fpr, tpr, color='navy')
    else:
        auc_score = 0

    if permutation_test:
        train_indices, test_indices = partition_indices()
        custom_cv = [(train_indices, test_indices)]
        X = np.zeros((975, 1000))
        X[train_indices] = X_train
        X[test_indices] = X_test
        Y = np.zeros((975,))
        Y[train_indices] = Y_train
        Y[test_indices] = Y_test
        clf = LogisticRegression()
        score, perms, pval = permutation_test_score(clf, X, Y, cv=custom_cv, verbose=1)
        print(score, perms.mean(), pval)

    return test_acc_score, auc_score

def run_object_regression(image_file, event_file, mask_file, plot=False):
    events = pd.read_csv(event_file, index_col=0)
    events = events[events['onset'] >= 40.5]
    events['onset'] -= 40.5
    object_labels = events['trial_type'].unique()
    test_accuracies = []
    test_aucs = []
    num_successful = 0
    for obj in object_labels:
        print(obj)
        try:
            test_acc, auc = run_single_subject(image_file, event_file, mask_file, trial_type=obj, plot=plot)
            test_accuracies.append(test_acc)
            test_aucs.append(auc)
            num_successful += 1
        except ValueError as e:
            print('Failed: ' + str(e))
    print('Average accuracy: ' + str(np.mean(test_accuracies)))
    print('Average AUC: ' + str(np.mean(test_aucs)))
    print('Successful: %d / %d = %f' % (num_successful, len(object_labels), num_successful / float(len(object_labels))))

def run_analysis(image_files, event_file, mask_file, multiclass=False, permutation_test=False):
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
    if multiclass:
        labels = get_multiclass_labels(event_file).as_matrix().T[0]
    else:
        labels = np.around(get_labels(event_file, 975).as_matrix()).T[0]
    Y_train = np.concatenate([labels] * 10)#(len(image_files) - 1))
    print('Labels shape: ' + str(Y_train.shape))
    print('Percentage of class 1: ' + str(max(np.unique(labels, return_counts=True)[1]) / float(len(labels))))

    # Reduce dimensionality
    dim_red = SelectKBest(f_classif, k=1000)
    X_train = dim_red.fit_transform(X_train, Y_train)
    X_test = dim_red.transform(X_test)
    print('Dimensionality reduced, new data shape: ' + str(X_train.shape))

    # Re-run classification
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    y_preds = clf.predict(X_test)
    test_acc_score = accuracy_score(labels, y_preds)
    print('Test accuracy score: ' + str(test_acc_score))

    if not multiclass:
        y_probs = clf.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(labels, y_probs)
        print('AUC score: ' + str(auc_score))

    if permutation_test:
        custom_cv = [(range(9750), range(9750, 9750 + 975))]
        X = np.concatenate([X_train, X_test])
        Y = np.concatenate([Y_train, labels])
        clf = LogisticRegression()
        score, perms, pval = permutation_test_score(clf, X, Y, cv=custom_cv, verbose=1)
        print(score, perms.mean(), pval)


if __name__ == '__main__':
    event_file = sys.argv[1]
    mask_file = sys.argv[2]
    image_files = sys.argv[3:]
    # run_analysis(image_files, event_file, mask_file, permutation_test=True)
    # run_single_subject(image_files[0], event_file, mask_file, permutation_test=True)
    run_object_regression(image_files[0], event_file, mask_file)
    # plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.show()
