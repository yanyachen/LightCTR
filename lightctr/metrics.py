import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt


def logloss(prediction, label):
    prediction = np.nan_to_num(prediction, copy=False)
    prediction = prediction.astype('float64', copy=False)
    eps = 1e-15
    prediction = np.fmax(np.fmin(prediction, 1.0 - eps), eps)
    logloss = -np.mean(
        label * np.log(prediction) + (1.0 - label) * np.log(1.0 - prediction)
    )
    return logloss


def normalized_cross_entropy(prediction, label):
    nce = logloss(prediction, label) / logloss(label.mean(), label)
    return nce


def auc(prediction, label):
    prediction_rank = rankdata(prediction, method='average')
    n_pos = np.sum(label == 1.0)
    n_neg = np.sum(label == 0.0)
    auc = (
        np.sum(prediction_rank[label == 1.0]) - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)
    return auc


def copc_plot(prediction, label, num_bins, title='Click on Predicted Click'):
    # Compute
    df_bin = (pd.DataFrame({'prediction': prediction, 'label': label})
                .assign(bin=lambda df: pd.qcut(
                    df['prediction'], q=num_bins, duplicates='drop'))
                .groupby(by='bin')
                .mean())
    # Scatter Plot
    plt.scatter(x='prediction', y='label', data=df_bin)
    # Diagonal Line
    plt.plot([0, 1], [0, 1], '-')
    # Limits
    value_max = df_bin.values.max()
    plt.xlim(0, value_max*1.07)
    plt.ylim(0, value_max*1.07)
    # Label
    plt.title(title)
    plt.xlabel('prediction')
    plt.ylabel('label')
    # Show
    plt.show()
