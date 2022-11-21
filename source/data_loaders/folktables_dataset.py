from river import stream
from river.datasets import base


class FolktablesDataset(base.FileDataset):
    def __init__(self, directory="../datasets", filename="folktables-NY-2018.csv", delimiter=','):
        super().__init__(
            filename=filename,
            directory=directory,
            n_features=16,
            task=base.BINARY_CLF,
        )
        self.delimiter = delimiter

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="ESR",
            delimiter=self.delimiter,
            converters={
                "AGEP": int,
                "SCHL": int,
                "MAR": str,
                "RELP": str,
                "DIS": str,
                "ESP": str,
                "CIT": str,
                "MIG": str,
                "MIL": str,
                "ANC": str,
                "NATIVITY": str,
                "DEAR": str,
                "DEYE": str,
                "DREM": str,
                "SEX": str,
                "RAC1P": str,
                "ESR": int,
            },
        )
