import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from river import utils
from random import Random

from source.config import SEED
from source.utils.stability_utils import count_prediction_stats, get_per_sample_accuracy


class StabilityAnalyzer:
    def __init__(self, base_model, n_estimators=10, batch_size=100):
        """
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]

        self.w = 6
        self._rng = Random(SEED)
        self.n_sample = 0
        self.batch_size = batch_size
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

        # Metrics history
        self.accuracy_lst = []
        self.mean_lst = []
        self.std_lst = []
        self.iqr_lst = []
        self.per_sample_accuracy_lst = []
        self.label_stability_lst = []

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
        if self.n_sample % self.batch_size != 0:
            self.sample_batch.append(x)
            self.y_true_lst.append(y_true)
            return

        # Quantify uncertainty for the bet model
        self.UQ_by_online_bagging(verbose=False)
        self.print_metrics()

        self.sample_batch = []
        self.y_true_lst = []

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

        # TODO: add sync with true model
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

        # Save metrics history
        self.accuracy_lst.append(self.accuracy)
        self.mean_lst.append(self.mean)
        self.std_lst.append(self.std)
        self.iqr_lst.append(self.iqr)
        self.per_sample_accuracy_lst.append(self.per_sample_accuracy)
        self.label_stability_lst.append(self.label_stability)

    def print_metrics(self):
        print(f'Sample number: {self.n_sample}\n'
              f'Accuracy: {self.accuracy}\n'
              f'Mean: {self.mean}\n'
              f'Std: {self.std}\n'
              f'IQR: {self.iqr}\n'
              f'Per sample accuracy: {self.per_sample_accuracy}\n'
              f'Label stability: {self.label_stability}\n\n')

    def plot_metrics_history(self):
        x_ticks = [(n_metric + 1) * self.batch_size for n_metric in range(len(self.label_stability_lst))]

        sns.set(rc={'figure.figsize':(15, 5)})

        # Plot the Accuracy history
        for label, metrics_lst in [('Accuracy', self.accuracy_lst), ('Mean', self.mean_lst), ('Std', self.std_lst),
                                   ('IQR', self.iqr_lst), ('Per sample accuracy', self.per_sample_accuracy_lst),
                                   ('Label stability', self.label_stability_lst)]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.grid(alpha=0.75)
            ax.plot(x_ticks, metrics_lst, lw=3, color='blue', alpha=0.8, label=label)
            ax.set_title(f'{label} {round(metrics_lst[-1], 4)}')
            plt.xlabel("Sample number")
            plt.ylabel(label)

        plt.show()
