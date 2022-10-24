import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from river import metrics


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def evaluate_binary_model(dataset, model, measure_every=1000):
    cm = metrics.ConfusionMatrix()
    acc_metric = metrics.Accuracy()
    # Weighted-average F1 score.
    # This works by computing the F1 score per class,
    # and then performs a global weighted average by using the support of each class.
    f1_metric = metrics.WeightedF1()

    acc_metrics = []
    f1_metrics = []
    for idx, (x, y_true) in enumerate(dataset):
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)

        # Update the error metric
        if y_pred is not None:
            cm = cm.update(y_true, y_pred)
            acc_metric = acc_metric.update(y_true, y_pred)
            f1_metric = f1_metric.update(y_true, y_pred)
        if (idx + 1) % measure_every == 0:
            acc_metrics.append(acc_metric.get())
            f1_metrics.append(f1_metric.get())
            print(f'Index: {idx + 1}; {acc_metric}; {f1_metric}')

        model = model.learn_one(x=x, y=y_true)

    print('\n\n\nDebug pipeline:\n' + model.debug_one(x))

    sns.set(rc={'figure.figsize':(15, 5)})

    # Plot the Accuracy results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(acc_metrics, lw=3, color='#2ecc71', alpha=0.8, label='Accuracy')
    ax.set_title(acc_metric)

    # Plot the F1 results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(f1_metrics, lw=3, color='#e74c3c', alpha=0.8, label='F1')
    ax.set_title(f1_metric)

    # Plot the F1 results
    fig, ax = plt.subplots(figsize=(10, 8))
    print('\n\nConfusion matrix:\n', cm)
    cm_dict = ddict2dict(cm.data)
    print('\nConfusion matrix dict:\n', cm_dict)
    ax = sns.heatmap(pd.DataFrame(cm_dict).T, annot=True, fmt=".0f")
    ax.set(xlabel="Predicted label", ylabel="True label")
    plt.show()
