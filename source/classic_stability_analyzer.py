import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy

from source.config import BOOTSTRAP_FRACTION
from source.utils.simple_utils import get_logger
from source.utils.EDA_utils import plot_generic
from source.utils.stability_utils import generate_bootstrap
from source.utils.stability_utils import count_prediction_stats, get_per_sample_accuracy


class ClassicStabilityAnalyzer:
    def __init__(self, base_model, base_model_name, train_pd_dataset, test_pd_dataset, test_y_true,
                 target_column, dataset_name, n_estimators=100):
        """
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.dataset_name = dataset_name
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]

        self.__logger = get_logger()

        self.train_pd_dataset = train_pd_dataset
        self.test_pd_dataset = test_pd_dataset
        self.test_y_true = test_y_true.values
        self.target_column = target_column

        # Metrics
        self.general_accuracy = None
        self.mean = None
        self.std = None
        self.iqr = None
        self.per_sample_accuracy = None
        self.label_stability = None
        self.jitter = None

    def _batch_predict(self, classifier, test_df):
        X_test, y_test = self._get_features_target_split(test_df)
        return classifier.predict(X_test)

    def measure_stability_metrics(self, make_plots=False):
        """
        Measure metrics for the evaluation model. Display plots for analysis if needed. Save results to a .pkl file

        :param make_plots: bool, if display plots for analysis
        """
        # For computing fairness-related metrics
        # boostrap_size = int(BOOTSTRAP_FRACTION * self.train_pd_dataset.shape[0])
        boostrap_size = int(0.9 * self.train_pd_dataset.shape[0])

        # Quantify uncertainty for the bet model
        models_predictions = self.UQ_by_boostrap(boostrap_size, with_replacement=True)

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

        self.save_metrics_to_file()

    def UQ_by_boostrap(self, boostrap_size, with_replacement):
        """
        Quantifying uncertainty of predictive model by constructing an ensemble from bootstrapped samples
        """
        models_predictions = {idx: [] for idx in range(self.n_estimators)}
        for idx in range(self.n_estimators):
            self.__logger.info(f'Start testing of classifier {idx + 1} / {self.n_estimators}')
            classifier = self.models_lst[idx]
            df_sample = generate_bootstrap(self.train_pd_dataset, boostrap_size, with_replacement)
            classifier = self._fit_model(classifier, df_sample)
            models_predictions[idx] = self._batch_predict(classifier, self.test_pd_dataset)
            self.__logger.info(f'Classifier {idx + 1} / {self.n_estimators} was tested')

        return models_predictions

    def _get_features_target_split(self, df):
        y = df[self.target_column]
        X = df.drop([self.target_column], axis=1)
        return X, y

    def _fit_model(self, classifier, train_df):
        X_train, y_train = self._get_features_target_split(train_df)
        return classifier.fit(X_train, y_train)

    def __update_metrics(self, accuracy, means, stds, iqr, per_sample_accuracy, label_stability, jitter):
        self.general_accuracy = np.round(accuracy, 4)
        self.mean = np.round(np.mean(means), 4)
        self.std = np.round(np.mean(stds), 4)
        self.iqr = np.round(np.mean(iqr), 4)
        self.per_sample_accuracy = np.round(np.mean(per_sample_accuracy), 4)
        self.label_stability = np.round(np.mean(label_stability), 4)
        self.jitter = np.round(jitter, 4)

    def print_metrics(self):
        print(f'General Ensemble Accuracy: {self.general_accuracy}\n'
              f'Mean: {self.mean}\n'
              f'Std: {self.std}\n'
              f'IQR: {self.iqr}\n'
              f'Per sample accuracy: {self.per_sample_accuracy}\n'
              f'Label stability: {self.label_stability}\n'
              f'Jitter: {self.jitter}\n\n')

    def save_metrics_to_file(self):
        metrics_to_report = {}
        metrics_to_report['General_Ensemble_Accuracy'] = [self.general_accuracy]
        metrics_to_report['Mean'] = [self.mean]
        metrics_to_report['Std'] = [self.std]
        metrics_to_report['IQR'] = [self.iqr]
        metrics_to_report['Per_Sample_Accuracy'] = [self.per_sample_accuracy]
        metrics_to_report['Label_Stability'] = [self.label_stability]
        metrics_to_report['Jitter'] = [self.jitter]
        metrics_df = pd.DataFrame(metrics_to_report)

        dir_path = os.path.join('..', '..', 'results', 'models_stability_metrics')
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{self.dataset_name}_{self.n_estimators}_estimators_{self.base_model_name}_base_model_stability_metrics.csv"
        metrics_df.to_csv(f'{dir_path}/{filename}', index=False)
