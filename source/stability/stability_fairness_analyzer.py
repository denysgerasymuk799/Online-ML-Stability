import os
import pandas as pd

from source.stability.fairness_analyzer import FairnessAnalyzer
from source.stability.base_stability_analyzer import BaseStabilityAnalyzer


class StabilityFairnessAnalyzer:
    def __init__(self, stability_analyzer: BaseStabilityAnalyzer, test_groups: dict):
        self.dataset_name = stability_analyzer.dataset_name
        self.n_estimators = stability_analyzer.n_estimators
        self.base_model_name = stability_analyzer.base_model_name

        self.__stability_analyzer = stability_analyzer
        self.__fairness_analyzer = FairnessAnalyzer(test_groups)
        self.stability_metrics_dct = dict()
        self.fairness_metrics_dct = dict()

    def measure_metrics(self, make_plots=True):
        """
        Measure metrics for the base model. Display stability plots for analysis if needed. Save results to a .pkl file

        :param make_plots: bool, if display plots for analysis
        """
        y_preds, test_y_true = self.__stability_analyzer.measure_stability_metrics(make_plots, save_results=False)
        self.stability_metrics_dct = self.__stability_analyzer.get_metrics_dict()

        # Count and display fairness metrics
        self.fairness_metrics_dct = self.__fairness_analyzer.get_metrics_dict(y_preds, test_y_true)

        # Save results to a .pkl file
        self.save_metrics_to_file()

    def save_metrics_to_file(self, save_dir_path=os.path.join('..', '..', 'results', 'models_stability_metrics')):
        metrics_to_report = {}
        metrics_to_report['Dataset_Name'] = [self.dataset_name]
        metrics_to_report['Base_Model_Name'] = [self.base_model_name]
        metrics_to_report['N_Estimators'] = [self.n_estimators]

        # Add stability metrics to metrics_to_report
        for stability_metric_key in self.stability_metrics_dct.keys():
            metrics_to_report[stability_metric_key] = [self.stability_metrics_dct[stability_metric_key]]

        # Add fairness metrics to metrics_to_report
        for fairness_metric_key in self.fairness_metrics_dct.keys():
            if fairness_metric_key == 'Statistical_Parity_Difference':
                fairness_metric_name = 'SPD'
            elif fairness_metric_key == 'Equal_Opportunity':
                fairness_metric_name = 'EO'
            else:
                fairness_metric_name = fairness_metric_key

            for group_type in ['Race', 'Sex', 'Race_Sex']:
                metrics_to_report[f'{fairness_metric_name}_{group_type}'] = [
                    self.fairness_metrics_dct[fairness_metric_key][group_type].loc[fairness_metric_key]
                ]

        metrics_df = pd.DataFrame(metrics_to_report)
        os.makedirs(save_dir_path, exist_ok=True)

        filename = f"{self.dataset_name}_{self.n_estimators}_estimators_{self.base_model_name}_base_model_stability_fairness_metrics.csv"
        metrics_df.to_csv(f'{save_dir_path}/{filename}', index=False)
