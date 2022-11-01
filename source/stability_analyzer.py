import numpy as np

from copy import deepcopy
from river import utils
from random import Random

from source.config import SEED
from source.utils.EDA_utils import plot_generic
from source.utils.stability_analysis import count_prediction_stats, get_per_sample_accuracy


class StabilityAnalyzer:
    def __init__(self, base_model, n_estimators=10, metric_memory_length=100):
        """
        :param X_data_tpl: a tuple of X_train_features and X_test_features; used to fit and test evaluation_model
        :param y_data_tpl: a tuple of y_train and y_test; used for evaluation_model
        :param test_groups: advantage and disadvantage groups to measure fairness metrics
        :param evaluation_model: the best model for the dataset; fairness and stability metrics will be measure for it
        :param imputation_technique: a name of imputation technique to name result .pkl file with metrics
        :param null_scenario_name: a name of null simulation method; just used to name a result .pkl file
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]

        self.w = 6
        self._rng = Random(SEED)
        self.n_sample = 0
        self.metric_memory_length = metric_memory_length
        self.y_true_lst = []
        self.model_predictions = {idx: [] for idx in range(self.n_estimators)}
        self.skip_first = True

        # Metrics
        self.accuracy = None
        self.mean = None
        self.std = None
        self.iqr = None
        self.per_sample_accuracy = None
        self.label_stability = None

    def measure_metrics(self, x, y_true, make_plots=False):
        """
        Measure metrics for the evaluation model. Display plots for analysis if needed. Save results to a .pkl file

        :param make_plots: bool, if display plots for analysis
        """

        # Quantify uncertainty for the bet model
        self.UQ_by_online_bagging(x, y_true, verbose=False)

        if self.skip_first:
            self.skip_first = False
            return
        else:
            self.n_sample += 1
            self._rolling_update(self.y_true_lst, y_true)

        # Count metrics
        y_preds, results, means, stds, iqr, accuracy = count_prediction_stats(self.y_true_lst,
                                                                              uq_results=self.model_predictions)
        per_sample_accuracy, label_stability = get_per_sample_accuracy(self.y_true_lst, results)

        # Display plots if needed
        if make_plots:
            plot_generic(means, stds, "Mean of probability", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Probability mean vs Standard deviation")
            plot_generic(stds, label_stability, "Standard deviation", "Label stability", x_lim=0.5, y_lim=1.01, plot_title="Standard deviation vs Label stability")
            plot_generic(means, label_stability, "Mean", "Label stability", x_lim=1.01, y_lim=1.01, plot_title="Mean vs Label stability")
            plot_generic(per_sample_accuracy, stds, "Accuracy", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Accuracy vs Standard deviation")
            plot_generic(per_sample_accuracy, iqr, "Accuracy", "Inter quantile range", x_lim=1.01, y_lim=1.01, plot_title="Accuracy vs Inter quantile range")

        self.__update_metrics(accuracy, means, stds, iqr, per_sample_accuracy, label_stability)
        if self.n_sample <= 5:
            print(f'y_true_lst: {self.y_true_lst}\nmodel_predictions: {self.model_predictions}\n\n')
            self.print_metrics()

    def UQ_by_online_bagging(self, x, y_true, verbose=True):
        """
        Quantifying uncertainty of predictive model by constructing an ensemble from bootstrapped samples
        """
        for idx in range(self.n_estimators):
            classifier = self.models_lst[idx]
            y_pred = classifier.predict_one(x)
            # TODO: not sure if it learns here
            # print(f'Before training\nx={x}\ny={y_true}')
            self.models_lst[idx] = classifier.learn_one(x=x, y=y_true)

            if y_pred is not None:
                self._rolling_update(self.model_predictions[idx], y_pred)

            # if verbose:
            #     print(idx)
            #     print("Train acc:", model.score(X_sample, y_sample))
            #     print("Val acc:", model.score(self.X_test_imputed, self.y_test))

    def UQ_by_online_bagging_v2(self, x, y_true, verbose=True):
        """
        Quantifying uncertainty of predictive model by constructing an ensemble from bootstrapped samples
        """
        for idx in range(self.n_estimators):
            classifier = self.models_lst[idx]
            y_pred = classifier.predict_one(x)
            # TODO: not sure if it learns here
            # print(f'Before training\nx={x}\ny={y_true}')
            k = self._leveraging_bag(x=x, y=y_true)
            for _ in range(k):
                self.models_lst[idx] = classifier.learn_one(x=x, y=y_true)

            if y_pred is not None:
                self._rolling_update(self.model_predictions[idx], y_pred)

    def _rolling_update(self, lst, item):
        if self.n_sample <= self.metric_memory_length:
            lst.append(item)
        else:
            lst[self.n_sample % self.metric_memory_length] = item

    def _leveraging_bag(self, **kwargs):
        # Leveraging bagging
        return utils.random.poisson(self.w, self._rng)

    def __update_metrics(self, accuracy, means, stds, iqr, per_sample_accuracy, label_stability):
        self.accuracy = accuracy
        self.mean = np.mean(means)
        self.std = np.mean(stds)
        self.iqr = np.mean(iqr)
        self.per_sample_accuracy = np.mean(per_sample_accuracy)
        self.label_stability = np.mean(label_stability)

    def print_metrics(self):
        print(f'Accuracy: {self.accuracy}\n'
              f'Mean: {self.mean}\n'
              f'Std: {self.std}\n'
              f'IQR: {self.iqr}\n'
              f'Per sample accuracy: {self.per_sample_accuracy}\n'
              f'Label stability: {self.label_stability}\n\n')
