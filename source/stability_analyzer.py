import numpy as np

from copy import deepcopy
from river import utils
from random import Random

from source.config import SEED
from source.utils.EDA_utils import plot_generic
from source.utils.stability_utils import count_prediction_stats, get_per_sample_accuracy


class StabilityAnalyzer:
    def __init__(self, base_model, n_estimators=10, metric_memory_length=100):
        """
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
        # self.model_predictions = {idx: [] for idx in range(self.n_estimators)}
        self.sample_batch = []

        # Metrics
        self.accuracy = None
        self.mean = None
        self.std = None
        self.iqr = None
        self.per_sample_accuracy = None
        self.label_stability = None

    @staticmethod
    def _batch_predict(classifier, sample_batch):
        return [classifier.predict_one(x) for x in sample_batch]

    def measure_stability_metrics(self, x, y_true, make_plots=False):
        """
        Measure metrics for the evaluation model. Display plots for analysis if needed. Save results to a .pkl file

        :param make_plots: bool, if display plots for analysis
        """

        # TODO: add a sliding window for rapid-updates or rolling update of a sample batch

        self.n_sample += 1
        if self.n_sample % self.metric_memory_length != 0:
            self.sample_batch.append(x)
            self.y_true_lst.append(y_true)
            return

        print('n_sample: ', self.n_sample)

        # Quantify uncertainty for the bet model
        self.UQ_by_online_bagging(verbose=False)
        self.print_metrics()

        self.sample_batch = []
        self.y_true_lst = []

        # # Display plots if needed
        # if make_plots:
        #     plot_generic(means, stds, "Mean of probability", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Probability mean vs Standard deviation")
        #     plot_generic(stds, label_stability, "Standard deviation", "Label stability", x_lim=0.5, y_lim=1.01, plot_title="Standard deviation vs Label stability")
        #     plot_generic(means, label_stability, "Mean", "Label stability", x_lim=1.01, y_lim=1.01, plot_title="Mean vs Label stability")
        #     plot_generic(per_sample_accuracy, stds, "Accuracy", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Accuracy vs Standard deviation")
        #     plot_generic(per_sample_accuracy, iqr, "Accuracy", "Inter quantile range", x_lim=1.01, y_lim=1.01, plot_title="Accuracy vs Inter quantile range")

    def UQ_by_online_bagging(self, verbose=True):
        """
        Quantifying uncertainty of predictive model by constructing an ensemble from bootstrapped samples
        """
        models_predictions = {idx: [] for idx in range(self.n_estimators)}
        for idx in range(self.n_estimators):
            classifier = self.models_lst[idx]
            models_predictions[idx] = StabilityAnalyzer._batch_predict(classifier, self.sample_batch)

        # Count metrics
        y_preds, results, means, stds, iqr, accuracy = count_prediction_stats(self.y_true_lst,
                                                                              uq_results=models_predictions)
        per_sample_accuracy, label_stability = get_per_sample_accuracy(self.y_true_lst, results)
        self.__update_metrics(accuracy, means, stds, iqr, per_sample_accuracy, label_stability)

        # Sync with an original model and apply online bagging for future stability measurements
        # self._sync_with_true_model(true_model)
        self._models_fit_by_online_bagging()

    def _models_fit_by_online_bagging(self):
        for (x, y_true) in zip(self.sample_batch, self.y_true_lst):
            for idx in range(self.n_estimators):
                classifier = self.models_lst[idx]
                # TODO: not sure if it learns here
                # print(f'Before training\nx={x}\ny={y_true}')
                k = self._leveraging_bag(x=x, y=y_true)
                for _ in range(k):
                    self.models_lst[idx] = classifier.learn_one(x=x, y=y_true)

    def _sync_with_true_model(self, true_model):
        self.models_lst = [deepcopy(true_model) for _ in range(self.n_estimators)]

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
