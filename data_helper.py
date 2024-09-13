
from sklearn.model_selection import KFold
import torch
import numpy as np
import os


def preprocess_data_array(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds,shuffle=True)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    train_channel_means = np.mean(X_train, axis=(0,1,2))
    train_channel_std =   np.std(X_train, axis=(0,1,2))
    return X_train, X_test, train_indices , test_indices


def get_features(data):
    # Initialize an empty list to hold each view's data
    views = []

    # Iterate over the third dimension and collect each view
    for i in range(data.shape[2]):
        views.append(data[:, :, i])

    # Concatenate all views along the second axis
    fts_mat = np.concatenate(views, axis=1)

    return fts_mat



