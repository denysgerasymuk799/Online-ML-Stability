import pandas as pd

from source.utils.analysis_helper import TPR_diff, DisparateImpact, BaseRates, Accuracy_diff, StatisticalParity_diff


class FairnessAnalyzer:
    def __init__(self, test_groups):
        self.test_groups = test_groups
        self.fairness_metrics_dict = dict()
    
    def measure_metrics(self, y_preds, y_test):
        metrics = ['Accuracy', 'Disparate_Impact', 'Equal_Opportunity', 'Statistical_Parity_Difference']

        for metric in metrics:
            self.fairness_metrics_dict[metric] = self.compute_fairness_metric(y_preds, y_test, metric)
        self.print_metrics()

    def get_metrics_dict(self):
        return self.fairness_metrics_dict

    def print_metrics(self):
        print('\n')
        print("#" * 30, " Fairness metrics ", "#" * 30)
        for key in self.fairness_metrics_dict.keys():
            print('\n' + '#' * 20 + f' {key} ' + '#' * 20)
            print(self.fairness_metrics_dict[key])

    def compute_fairness_metric(self, predicted, true, metric):
        res = pd.DataFrame({})
        for group_name in self.test_groups.keys():
            group = self.test_groups[group_name]
            if metric == 'Equal_Opportunity':
                temp = TPR_diff(predicted, true, group['values'], group['advantaged'], group['disadvantaged'])

            if metric == 'Disparate_Impact':
                temp = DisparateImpact(predicted, true, group['values'], group['advantaged'], group['disadvantaged'])

            if metric == 'Base_Rate':
                temp = BaseRates(predicted, true, group['values'], group['advantaged'], group['disadvantaged'])

            if metric == 'Accuracy':
                temp = Accuracy_diff(predicted, true, group['values'], group['advantaged'], group['disadvantaged'])

            if metric == 'Statistical_Parity_Difference':
                temp = StatisticalParity_diff(predicted, true, group['values'], group['advantaged'], group['disadvantaged'])

            temp = [round(val, 4) for val in temp] # round values for easier understanding
            temp_df = pd.DataFrame(temp, columns=[group_name], index=['adv', 'disadv', metric])
            res = pd.concat([res, temp_df], axis=1)

        return res
