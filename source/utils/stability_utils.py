import itertools
import numpy as np
import pandas as pd
import scipy as sp

from matplotlib import pyplot as plt

from source.utils.simple_utils import set_size


def compute_label_stability(predicted_labels):
    '''
    Label stability is defined as the absolute difference between the number of times the sample is classified as 0 and 1
    If the absolute difference is large, the label is more stable
    If the difference is exactly zero then it's extremely unstable --- equally likely to be classified as 0 or 1
    '''
    count_pos = sum(predicted_labels)
    count_neg = len(predicted_labels) - count_pos
    return np.abs(count_pos - count_neg)/len(predicted_labels)


def compute_churn(predicted_labels_1, predicted_labels_2):
    return sum(int(predicted_labels_1[i] != predicted_labels_2[i])
               for i in range(len(predicted_labels_1))) / len(predicted_labels_1)


def compute_jitter(models_prediction_labels):
    n_models = len(models_prediction_labels)
    models_idx_lst = [i for i in range(n_models)]
    churns_sum = 0
    for i, j in itertools.combinations(models_idx_lst, 2):
        churns_sum += compute_churn(models_prediction_labels[i], models_prediction_labels[j])

    return churns_sum / (n_models * (n_models - 1) * 0.5)


def count_prediction_stats(y_test, uq_results):
    results = pd.DataFrame(uq_results).transpose()
    means = results.mean().values
    stds = results.std().values
    iqr = sp.stats.iqr(results, axis=0)
    jitter = compute_jitter(uq_results)

    # y_preds = np.array([int(x<0.5) for x in results.mean().values])
    y_preds = np.array([round(x) for x in results.mean().values])
    # print(f'y_preds: {y_preds}\ny_test: {y_test}\n')
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
        # per_sample_predictions[sample] =  [int(x<0.5) for x in results[sample].values]
        # TODO: is it correct to measure label stability in such a way
        label_stability.append(compute_label_stability(per_sample_predictions[sample]))

        if y_test[sample] == 1:
            acc = np.mean(per_sample_predictions[sample])
        elif y_test[sample] == 0:
            acc = 1 - np.mean(per_sample_predictions[sample])
        if acc is not None:
            per_sample_accuracy.append(acc)

    return per_sample_accuracy, label_stability


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
