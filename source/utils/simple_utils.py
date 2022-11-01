import matplotlib.pyplot as plt

from source.config import FOLKTABLES_COLUMN_TO_TYPE


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
