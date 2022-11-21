from river import stream
from river.datasets import base


class HealthRiskDataset(base.FileDataset):
    """
    Dataset loader for River incremental models
    """
    def __init__(self, directory="../datasets", filename="Maternal-Health-Risk-Data-Set.csv", delimiter=','):
        super().__init__(
            filename=filename,
            directory=directory,
            n_features=6,
            task=base.BINARY_CLF,
        )
        self.delimiter = delimiter

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="RiskLevel",
            delimiter=self.delimiter,
            converters={
                "Age": int,
                "SystolicBP": int,
                "DiastolicBP": int,
                "BS": float,
                "BodyTemp": float,
                "HeartRate": int,
            },
        )
