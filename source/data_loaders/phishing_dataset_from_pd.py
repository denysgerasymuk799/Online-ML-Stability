from source.utils.river_utils import iter_pd_dataset


class PhishingDatasetFromPandas:
    """
    Loader that converts a pandas df to River dataset for incremental models
    """
    def __init__(self, pd_dataset):
        self.pd_dataset = pd_dataset

    def __iter__(self):
        return iter_pd_dataset(
            self.pd_dataset,
            target="is_phishing",
            converters={
                "empty_server_form_handler": float,
                "popup_window": float,
                "https": float,
                "request_from_other_domain": float,
                "anchor_from_other_domain": float,
                "is_popular": float,
                "long_url": float,
                "age_of_domain": int,
                "ip_in_url": int,
                "is_phishing": lambda x: 1 if x == "1" else 0,
            },
        )
