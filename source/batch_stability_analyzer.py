from source.config import TRAIN_CV_BOOTSTRAP_FRACTION
from source.base_stability_analyzer import BaseStabilityAnalyzer


class BatchStabilityAnalyzer(BaseStabilityAnalyzer):
    def __init__(self, base_model, base_model_name, train_pd_dataset, test_pd_dataset, test_y_true,
                 target_column, dataset_name, n_estimators=100):
        """
        :param n_estimators: a number of estimators in ensemble to measure evaluation_model stability
        """
        super().__init__(base_model, base_model_name, TRAIN_CV_BOOTSTRAP_FRACTION,
                         train_pd_dataset, test_pd_dataset, test_y_true, dataset_name, n_estimators)
        self.target_column = target_column

    def _batch_predict(self, classifier, test_df):
        X_test, y_test = self._get_features_target_split(test_df)
        return classifier.predict(X_test)

    def _get_features_target_split(self, df):
        y = df[self.target_column]
        X = df.drop([self.target_column], axis=1)
        return X, y

    def _fit_model(self, classifier, train_df):
        X_train, y_train = self._get_features_target_split(train_df)
        return classifier.fit(X_train, y_train)
