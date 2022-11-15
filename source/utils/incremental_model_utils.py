from river import metrics

from source.config import logger


def train_incremental_model(base_model, dataset, dataset_size, label_mapping, train_fraction=1.0):
    # Conduct model training
    train_size = int(dataset_size * train_fraction)
    acc_metric = metrics.Accuracy()
    f1_metric = metrics.WeightedF1()
    for idx, (x, y_true) in enumerate(dataset):
        y_true = label_mapping[y_true]
        y_pred = base_model.predict_one(x)

        # Update the error metric
        if y_pred is not None:
            acc_metric = acc_metric.update(y_true, y_pred)
            f1_metric = f1_metric.update(y_true, y_pred)

        base_model.learn_one(x=x, y=y_true)
        if (idx + 1) % 1000 == 0:
            logger.info(f'{idx + 1} iterations from {dataset_size} are passed')

        if idx + 1 == train_size:
            break

    print('Metrics after incremental model training:')
    print(acc_metric)
    print(f1_metric)

    return base_model
