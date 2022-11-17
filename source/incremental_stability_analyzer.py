from source.config import CV_BOOTSTRAP_FRACTION
from source.base_stability_analyzer import BaseStabilityAnalyzer


class IncrementalStabilityAnalyzer(BaseStabilityAnalyzer):
    def __init__(self, base_model, base_model_name, train_pd_dataset, test_pd_dataset, test_y_true,
                 dataset_reader, dataset_name, n_estimators=100, prediction_mapping=None):
        """
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        super().__init__(base_model, base_model_name, CV_BOOTSTRAP_FRACTION,
                         train_pd_dataset, dataset_reader(pd_dataset=test_pd_dataset),
                         test_y_true, dataset_name, n_estimators)
        self.__prediction_mapping = prediction_mapping
        self.dataset_reader = dataset_reader

    def _batch_predict(self, classifier, test_dataset):
        predictions = []
        for x, y_true in test_dataset:
            y_pred = classifier.predict_one(x)
            if self.__prediction_mapping is not None:
                y_pred = self.__prediction_mapping[y_pred]
            predictions.append(y_pred)

        return predictions

    def _fit_model(self, classifier, train_df):
        train_dataset = self.dataset_reader(pd_dataset=train_df)
        for x, y_true in train_dataset:
            classifier.learn_one(x=x, y=y_true)

        return classifier
