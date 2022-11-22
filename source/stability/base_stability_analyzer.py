import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy

from source.utils.simple_utils import get_logger
from source.utils.EDA_utils import plot_generic
from source.utils.stability_utils import generate_bootstrap
from source.utils.stability_utils import count_prediction_stats, get_per_sample_accuracy


class BaseStabilityAnalyzer:
    def __init__(self, base_model, base_model_name, bootstrap_fraction,
                 train_pd_dataset, test_pd_dataset, test_y_true, dataset_name, n_estimators=100):
        """
        :param base_model: base model for stability measuring
        :param base_model_name: model name like 'HoeffdingTreeClassifier' or 'LogisticRegression'
        :param bootstrap_fraction: [0-1], fraction from train_pd_dataset for fitting an ensemble of base models
        :param train_pd_dataset: pandas train dataset
        :param test_pd_dataset: pandas test dataset
        :param test_y_true: y value from test_pd_dataset
        :param dataset_name: str, like 'Folktables' or 'Phishing'
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.bootstrap_fraction = bootstrap_fraction
        self.dataset_name = dataset_name
        self.n_estimators = n_estimators
        self.models_lst = [deepcopy(base_model) for _ in range(n_estimators)]

        self.__logger = get_logger()

        self.train_pd_dataset = train_pd_dataset
        self.test_dataset = test_pd_dataset
        self.test_y_true = test_y_true.values

        # Metrics
        self.general_accuracy = None
        self.mean = None
        self.std = None
        self.iqr = None
        self.per_sample_accuracy = None
        self.label_stability = None
        self.jitter = None

    def _batch_predict(self, classifier, test_dataset):
        pass

    def measure_stability_metrics(self, make_plots=False, save_results=True):
        """
        Measure metrics for the base model. Display plots for analysis if needed. Save results to a .pkl file

        :param make_plots: bool, if display plots for analysis
        """
        # Quantify uncertainty for the base model
        boostrap_size = int(self.bootstrap_fraction * self.train_pd_dataset.shape[0])
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

        if save_results:
            self.save_metrics_to_file()
        else:
            return y_preds, self.test_y_true

    def UQ_by_boostrap(self, boostrap_size, with_replacement):
        """
        Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples
        """
        models_predictions = {idx: [] for idx in range(self.n_estimators)}
        for idx in range(self.n_estimators):
            self.__logger.info(f'Start testing of classifier {idx + 1} / {self.n_estimators}')
            classifier = self.models_lst[idx]
            df_sample = generate_bootstrap(self.train_pd_dataset, boostrap_size, with_replacement)
            classifier = self._fit_model(classifier, df_sample)
            models_predictions[idx] = self._batch_predict(classifier, self.test_dataset)
            self.__logger.info(f'Classifier {idx + 1} / {self.n_estimators} was tested')

        return models_predictions

    def _fit_model(self, classifier, train_df):
        pass

    def __update_metrics(self, accuracy, means, stds, iqr, per_sample_accuracy, label_stability, jitter):
        self.general_accuracy = np.round(accuracy, 4)
        self.mean = np.round(np.mean(means), 4)
        self.std = np.round(np.mean(stds), 4)
        self.iqr = np.round(np.mean(iqr), 4)
        self.per_sample_accuracy = np.round(np.mean(per_sample_accuracy), 4)
        self.label_stability = np.round(np.mean(label_stability), 4)
        self.jitter = np.round(jitter, 4)

    def print_metrics(self):
        print('\n')
        print("#" * 30, "Stability metrics", "#" * 30)
        print(f'General Ensemble Accuracy: {self.general_accuracy}\n'
              f'Mean: {self.mean}\n'
              f'Std: {self.std}\n'
              f'IQR: {self.iqr}\n'
              f'Per sample accuracy: {self.per_sample_accuracy}\n'
              f'Label stability: {self.label_stability}\n'
              f'Jitter: {self.jitter}\n\n')

    def get_metrics_dict(self):
        return {
            'General_Ensemble_Accuracy': self.general_accuracy,
            'Mean': self.mean,
            'Std': self.std,
            'IQR': self.iqr,
            'Per_Sample_Accuracy': self.per_sample_accuracy,
            'Label_Stability': self.label_stability,
            'Jitter': self.jitter,
        }

    def save_metrics_to_file(self):
        metrics_to_report = {}
        metrics_to_report['Dataset_Name'] = [self.dataset_name]
        metrics_to_report['Base_Model_Name'] = [self.base_model_name]
        metrics_to_report['N_Estimators'] = [self.n_estimators]

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
