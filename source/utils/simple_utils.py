import logging
import pandas as pd
import matplotlib.pyplot as plt

from source.config import FOLKTABLES_COLUMN_TO_TYPE
from source.custom_logger import CustomHandler


def get_logger():
    logger = logging.getLogger('root')
    logger.setLevel('INFO')
    logging.disable(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(CustomHandler())

    return logger


def get_folktables_column_type(column_name):
    for column_type in FOLKTABLES_COLUMN_TO_TYPE.keys():
        if column_name in FOLKTABLES_COLUMN_TO_TYPE[column_type]:
            return column_type
    return None


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def make_feature_df(data, categorical_columns, numerical_columns):
    """
    Return a dataset made by one-hot encoding for categorical columns and concatenate with numerical columns
    """
    feature_df = pd.get_dummies(data[categorical_columns], columns=categorical_columns)
    for col in numerical_columns:
        if col in data.columns:
            feature_df[col] = data[col]
    return feature_df
