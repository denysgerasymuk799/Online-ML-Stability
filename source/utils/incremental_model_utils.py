from river import metrics

from source.utils.simple_utils import get_logger


def train_incremental_model(base_model, dataset, dataset_size, print_every=1000,
                            label_mapping=None, prediction_mapping=None, train_fraction=1.0):
    logger = get_logger()

    # Conduct model training
    train_size = int(dataset_size * train_fraction)
    acc_metric = metrics.Accuracy()
    f1_metric = metrics.WeightedF1()
    for idx, (x, y_true) in enumerate(dataset):
        if label_mapping is not None:
            y_true = label_mapping[y_true]

        y_pred = base_model.predict_one(x)
        if prediction_mapping is not None:
            y_pred = prediction_mapping[y_pred]

        # Update the error metric
        if y_pred is not None:
            acc_metric = acc_metric.update(y_true, y_pred)
            f1_metric = f1_metric.update(y_true, y_pred)

        base_model.learn_one(x=x, y=y_true)
        if (idx + 1) % print_every == 0:
            logger.info(f'Iteration {idx + 1}/{train_size} -- {acc_metric}; {f1_metric}')

        if idx + 1 == train_size:
            break

    print('\n\nMetrics after incremental model training:')
    print(acc_metric)
    print(f1_metric)

    return base_model
