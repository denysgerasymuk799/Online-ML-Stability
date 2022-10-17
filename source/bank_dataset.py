from river import stream
from river.datasets import base


class BankDataset(base.FileDataset):
    def __init__(self):
        super().__init__(
            filename="bank-additional-full.csv",
            directory="../datasets",
            n_features=20,
            task=base.BINARY_CLF,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="y",
            delimiter=';',
            converters={
                "age": int,
                "duration": int,
                "campaign": int,
                "pdays": int,
                "previous": int,
                "emp.var.rate": float,
                "cons.price.idx": float,
                "cons.conf.idx": float,
                "euribor3m": float,
                "nr.employed": float
            },
        )