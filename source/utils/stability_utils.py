import os
import itertools
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt

from source.utils.simple_utils import set_size


def compute_label_stability(predicted_labels):
    """
    Label stability is defined as the absolute difference between the number of times the sample is classified as 0 and 1
    If the absolute difference is large, the label is more stable
    If the difference is exactly zero then it's extremely unstable --- equally likely to be classified as 0 or 1
    """
    count_pos = sum(predicted_labels)
    count_neg = len(predicted_labels) - count_pos
    return np.abs(count_pos - count_neg)/len(predicted_labels)


def compute_churn(predicted_labels_1, predicted_labels_2):
    """
    Pairwise stability metric for two model predictions
    """
    return sum(int(predicted_labels_1[i] != predicted_labels_2[i])
               for i in range(len(predicted_labels_1))) / len(predicted_labels_1)


def compute_jitter(models_prediction_labels):
    """
    Jitter is a stability metric that shows how the base model predictions fluctuate.
    Values closer to 0 -- perfect stability, values closer to 1 -- extremely bad stability.
    """
    n_models = len(models_prediction_labels)
    models_idx_lst = [i for i in range(n_models)]
    churns_sum = 0
    for i, j in itertools.combinations(models_idx_lst, 2):
        churns_sum += compute_churn(models_prediction_labels[i], models_prediction_labels[j])

    return churns_sum / (n_models * (n_models - 1) * 0.5)


def count_prediction_stats(y_test, uq_results):
    """
    Compute means, stds, iqr, accuracy, jitter and transform predictions to pd df

    :param y_test: true labels
    :param uq_results: predicted labels
    """
    results = pd.DataFrame(uq_results).transpose()
    means = results.mean().values
    stds = results.std().values
    iqr = sp.stats.iqr(results, axis=0)
    jitter = compute_jitter(uq_results)

    y_preds = np.array([round(x) for x in results.mean().values])
    accuracy = np.mean(np.array([y_preds[i] == int(y_test[i]) for i in range(len(y_test))]))

    return y_preds, results, means, stds, iqr, accuracy, jitter


def get_per_sample_accuracy(y_test, results):
    """

    :param y_test: y test dataset
    :param results: results variable from count_prediction_stats()
    :return: per_sample_accuracy and label_stability (refer to https://www.osti.gov/servlets/purl/1527311)
    """
    per_sample_predictions = {}
    label_stability = []
    per_sample_accuracy = []
    acc = None
    for sample in range(len(y_test)):
        per_sample_predictions[sample] =  [round(x) for x in results[sample].values]
        label_stability.append(compute_label_stability(per_sample_predictions[sample]))

        if y_test[sample] == 1:
            acc = np.mean(per_sample_predictions[sample])
        elif y_test[sample] == 0:
            acc = 1 - np.mean(per_sample_predictions[sample])
        if acc is not None:
            per_sample_accuracy.append(acc)

    return per_sample_accuracy, label_stability


def generate_bootstrap(df, boostrap_size, with_replacement=True):
    bootstrap_index = np.random.choice(df.shape[0], size=boostrap_size, replace=with_replacement)
    bootstrap_features = pd.DataFrame(df).iloc[bootstrap_index]
    if len(bootstrap_features) == boostrap_size:
        return bootstrap_features
    else:
        raise ValueError('Bootstrap samples are not of the size requested')


def display_result_plots(results_dir):
    sns.set_style("darkgrid")
    results = dict()
    filenames = [f for f in listdir(results_dir) if isfile(join(results_dir, f))]

    for filename in filenames:
        results_df = pd.read_csv(results_dir + filename)
        results[f'{results_df.iloc[0]["Base_Model_Name"]}_{results_df.iloc[0]["N_Estimators"]}_estimators'] = results_df

    y_metrics = ['SPD_Race', 'SPD_Sex', 'SPD_Race_Sex', 'EO_Race', 'EO_Sex', 'EO_Race_Sex']
    x_metrics = ['Label_Stability', 'General_Ensemble_Accuracy', 'Std']
    for x_metric in x_metrics:
        for y_metric in y_metrics:
            x_lim = 0.3 if x_metric == 'SD' else 1.0
            display_uncertainty_plot(results, x_metric, y_metric, x_lim)


def display_uncertainty_plot(results, x_metric, y_metric, x_lim):
    fig, ax = plt.subplots()
    set_size(15, 8, ax)

    # List of all markers -- https://matplotlib.org/stable/api/markers_api.html
    markers = ['.', 'o', '+', '*', '|', '<', '>', '^', 'v', '1', 's', 'x', 'D', 'P', 'H']
    techniques = results.keys()
    shapes = []
    for idx, technique in enumerate(techniques):
        a = ax.scatter(results[technique][x_metric], results[technique][y_metric], marker=markers[idx], s=100)
        shapes.append(a)

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.xlim(0, x_lim)
    plt.title(f'{x_metric} [{y_metric}]', fontsize=20)
    ax.legend(shapes, techniques, fontsize=12, title='Markers')

    plt.show()
