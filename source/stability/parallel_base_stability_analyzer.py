import numpy as np

from multiprocessing import cpu_count
from functools import partial
from multiprocessing import Pool

from source.utils.simple_utils import get_logger
from source.utils.stability_utils import generate_bootstrap
from source.stability.base_stability_analyzer import BaseStabilityAnalyzer


class ParallelBaseStabilityAnalyzer(BaseStabilityAnalyzer):
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
        self.__logger = get_logger()
        super().__init__(base_model, base_model_name, bootstrap_fraction,
                         train_pd_dataset, test_pd_dataset, test_y_true, dataset_name, n_estimators)

    def _UQ_task(self, train_dataset, test_dataset, boostrap_size, with_replacement, classifier):
        df_sample = generate_bootstrap(train_dataset, boostrap_size, with_replacement)
        classifier = self._fit_model(classifier, df_sample)

        return self._batch_predict(classifier, test_dataset)

    def UQ_by_boostrap(self, boostrap_size, with_replacement):
        """
        Quantifying uncertainty of the base model by constructing an ensemble from bootstrapped samples
        """
        self.__logger.info('Start model predictions')

        # Create the process pool
        with Pool(cpu_count() - 1) as pool:
            # Parallel model predictions
            models_predictions = np.array(
                pool.map(partial(self._UQ_task, self.train_pd_dataset, self.test_dataset,
                                 boostrap_size, with_replacement),
                         self.models_lst)
            )
        self.__logger.info(f'Generated predictions for {self.n_estimators} models')

        return models_predictions
