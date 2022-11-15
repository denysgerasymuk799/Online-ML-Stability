import random
import typing
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import StringIO

from river import base
from river import metrics
from river.stream import utils
from river.stream.iter_csv import DictReader


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def df_to_stream_buffer(df):
    buffer = StringIO()  # creating an empty buffer
    df.to_csv(buffer)  # filling that buffer
    buffer.seek(0) # set to the start of the stream

    return buffer


def iter_pd_dataset(
        pd_dataset,
        target: typing.Union[str, typing.List[str]] = None,
        converters: dict = None,
        parse_dates: dict = None,
        drop: typing.List[str] = None,
        drop_nones=False,
        fraction=1.0,
        seed: int = None,
        **kwargs,
) -> base.typing.Stream:

    buffer = df_to_stream_buffer(pd_dataset)
    for x in DictReader(fraction=fraction, rng=random.Random(seed), f=buffer, **kwargs):
        if drop:
            for i in drop:
                del x[i]

        # Cast the values to the given types
        if converters is not None:
            for i, t in converters.items():
                x[i] = t(x[i])

        # Drop Nones
        if drop_nones:
            for i in list(x):
                if x[i] is None:
                    del x[i]

        # Parse the dates
        if parse_dates is not None:
            for i, fmt in parse_dates.items():
                x[i] = dt.datetime.strptime(x[i], fmt)

        # Separate the target from the features
        y = None
        if isinstance(target, list):
            y = {name: x.pop(name) for name in target}
        elif target is not None:
            y = x.pop(target)

        yield x, y


def evaluate_binary_model(dataset, model, measure_every=1000, dataset_limit=None):
    cm = metrics.ConfusionMatrix()
    acc_metric = metrics.Accuracy()
    kappa_metric = metrics.CohenKappa()
    # Weighted-average F1 score.
    # This works by computing the F1 score per class,
    # and then performs a global weighted average by using the support of each class.
    f1_metric = metrics.WeightedF1()

    acc_metrics = []
    f1_metrics = []
    kappa_metrics = []
    for idx, (x, y_true) in enumerate(dataset):
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)

        # Update the error metric
        if y_pred is not None:
            cm = cm.update(y_true, y_pred)
            acc_metric = acc_metric.update(y_true, y_pred)
            kappa_metric = kappa_metric.update(y_true, y_pred)
            f1_metric = f1_metric.update(y_true, y_pred)
        if (idx + 1) % measure_every == 0:
            acc_metrics.append(acc_metric.get())
            kappa_metrics.append(kappa_metric.get())
            f1_metrics.append(f1_metric.get())
            print(f'Index: {idx + 1}; {acc_metric}; {kappa_metric}; {f1_metric}')

        # Shrink the dataset if needed
        if dataset_limit is not None and (idx + 1) == dataset_limit:
            break

        model = model.learn_one(x=x, y=y_true)

    print('\n\n\nDebug pipeline:\n' + model.debug_one(x))

    sns.set(rc={'figure.figsize':(15, 5)})

    # Plot the Accuracy results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(acc_metrics, lw=3, color='#2ecc71', alpha=0.8, label='Accuracy')
    ax.set_title(acc_metric)

    # Plot the Accuracy results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(kappa_metrics, lw=3, color='blue', alpha=0.8, label='CohenKappa')
    ax.set_title(kappa_metric)

    # Plot the F1 results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(f1_metrics, lw=3, color='#e74c3c', alpha=0.8, label='F1')
    ax.set_title(f1_metric)

    # Plot a confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    print('\n\nConfusion matrix:\n', cm)
    cm_dict = ddict2dict(cm.data)
    print('\nConfusion matrix dict:\n', cm_dict)
    ax = sns.heatmap(pd.DataFrame(cm_dict).T, annot=True, fmt=".0f")
    ax.set(xlabel="Predicted label", ylabel="True label")
    plt.show()
