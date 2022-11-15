from source.utils.river_utils import iter_pd_dataset


class FolktablesDatasetFromPandas:
    def __init__(self, pd_dataset):
        self.pd_dataset = pd_dataset

    def __iter__(self):
        return iter_pd_dataset(
            self.pd_dataset,
            target="ESR",
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
