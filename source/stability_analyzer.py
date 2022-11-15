import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from source.config import logger
from source.config import BOOTSTRAP_FRACTION
from source.utils.EDA_utils import plot_generic
from source.utils.stability_utils import generate_bootstrap
from source.utils.stability_utils import count_prediction_stats, get_per_sample_accuracy
from source.folktables_dataset_from_pd import FolktablesDatasetFromPandas


class StabilityAnalyzer:
    def __init__(self, base_model, train_pd_dataset, test_pd_dataset, test_y_true, n_estimators=100):
        """
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]
        self.__logger = logger

        self.train_pd_dataset = train_pd_dataset
        self.test_pd_dataset = test_pd_dataset
        self.test_dataset = FolktablesDatasetFromPandas(pd_dataset=self.test_pd_dataset)
        self.test_y_true = test_y_true.values

        # Metrics
        self.accuracy = None
        self.mean = None
        self.std = None
        self.iqr = None
        self.per_sample_accuracy = None
        self.label_stability = None
        self.jitter = None

    @staticmethod
    def _batch_predict(classifier, test_dataset):
        predictions = []
        for x, y_true in test_dataset:
            predictions.append(classifier.predict_one(x))

        return predictions

    def measure_stability_metrics(self, make_plots=False):
        """
        Measure metrics for the evaluation model. Display plots for analysis if needed. Save results to a .pkl file

        :param make_plots: bool, if display plots for analysis
        """
        # For computing fairness-related metrics
        boostrap_size = int(BOOTSTRAP_FRACTION * self.train_pd_dataset.shape[0])

        # Quantify uncertainty for the bet model
        models_predictions = self.UQ_by_boostrap(boostrap_size, with_replacement=True, verbose=False)

        # Count metrics
        y_preds, results, means, stds, iqr, accuracy, jitter = count_prediction_stats(self.test_y_true,
                                                                                      uq_results=models_predictions)
        per_sample_accuracy, label_stability = get_per_sample_accuracy(self.test_y_true, results)
        self.__update_metrics(accuracy, means, stds, iqr, per_sample_accuracy, label_stability, jitter)

        self.print_metrics()

        # Display plots if needed
        if make_plots:
            plot_generic(means, stds, "Mean of probability", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Probability mean vs Standard deviation")
            plot_generic(stds, label_stability, "Standard deviation", "Label stability", x_lim=0.5, y_lim=1.01, plot_title="Standard deviation vs Label stability")
            plot_generic(means, label_stability, "Mean", "Label stability", x_lim=1.01, y_lim=1.01, plot_title="Mean vs Label stability")
            plot_generic(per_sample_accuracy, stds, "Accuracy", "Standard deviation", x_lim=1.01, y_lim=0.5, plot_title="Accuracy vs Standard deviation")
            plot_generic(per_sample_accuracy, iqr, "Accuracy", "Inter quantile range", x_lim=1.01, y_lim=1.01, plot_title="Accuracy vs Inter quantile range")

    def UQ_by_boostrap(self, boostrap_size, with_replacement, verbose=True):
        """
        Quantifying uncertainty of predictive model by constructing an ensemble from bootstrapped samples
        """
        models_predictions = {idx: [] for idx in range(self.n_estimators)}
        for idx in range(self.n_estimators):
            self.__logger.info(f'Start testing of classifier {idx + 1} / {self.n_estimators}')
            classifier = self.models_lst[idx]
            df_sample = generate_bootstrap(self.train_pd_dataset, boostrap_size, with_replacement)
            classifier = self._fit_model(classifier, df_sample)
            models_predictions[idx] = StabilityAnalyzer._batch_predict(classifier, self.test_dataset)
            self.__logger.info(f'Classifier {idx + 1} / {self.n_estimators} was tested')

        return models_predictions

    def _fit_model(self, classifier, train_df):
        train_dataset = FolktablesDatasetFromPandas(pd_dataset=train_df)
        for x, y_true in train_dataset:
            classifier.learn_one(x=x, y=y_true)

        return classifier

    def __update_metrics(self, accuracy, means, stds, iqr, per_sample_accuracy, label_stability, jitter):
        self.accuracy = accuracy
        self.mean = np.mean(means)
        self.std = np.mean(stds)
        self.iqr = np.mean(iqr)
        self.per_sample_accuracy = np.mean(per_sample_accuracy)
        self.label_stability = np.mean(label_stability)
        self.jitter = jitter

    def print_metrics(self):
        print(f'Avg Classifiers Individual Accuracy: {self.accuracy}\n'
              f'Mean: {self.mean}\n'
              f'Std: {self.std}\n'
              f'IQR: {self.iqr}\n'
              f'Per sample accuracy: {self.per_sample_accuracy}\n'
              f'Label stability: {self.label_stability}\n'
              f'Jitter: {self.jitter}\n\n')
