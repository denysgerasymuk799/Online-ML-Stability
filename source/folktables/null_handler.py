import numpy as np
from scipy import stats


COLUMN_TO_TYPE = {
    "categorical": ['SCHL', 'MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC'],
    "numerical": ['AGEP']
}


def get_column_type(column_name):
    for column_type in COLUMN_TO_TYPE.keys():
        if column_name in COLUMN_TO_TYPE[column_type]:
            return column_type
    return None


def handle_df_nulls(input_data, how, column_names, condition_column=None):
    """
    Description: Processes the null values in the dataset
    Input:
    data: dataframe with missing values
    how: processing method, currently supports
        - 'special': corresponds to 'not applicable' scenario, designates null values as their own special category
        - 'impute-by-mode' : impute nulls by mode of the column values without nulls
        - 'impute-by-mode-trimmed' : the same as 'impute-by-mode', but the column is filtered from nulls,
        sorted in descending order, and top and bottom k% are removed from it. After that 'impute-by-mode' logic is applied
        - 'impute-by-mean' : impute nulls by mean of the column values without nulls
        - 'impute-by-mean-trimmed' : the same as 'impute-by-mean', but the column is filtered from nulls,
        sorted in descending order, and top and bottom k% are removed from it. After that 'impute-by-mean' logic is applied
        - 'impute-by-median' : impute nulls by median of the column values without nulls
        - 'impute-by-median-trimmed' : the same as 'impute-by-median', but the column is filtered from nulls,
        sorted in descending order, and top and bottom k% are removed from it. After that 'impute-by-median' logic is applied
        - 'impute-by-(mode/mean/median)-conditional' : the same as 'impute-by-(mode/mean/median)',
        but (mode/mean/median) is counted for each group and each group is imputed with this appropriate (mode/mean/median).
        Groups are created based on split of a dataset by RAC1P or SEX
    column-names: list of column names, for which the particular techniques needs to be applied

    Output:
    a dataframe with processed nulls
    """
    data = input_data.copy(deep=True)

    get_impute_value = None
    if how == 'special':
        get_impute_value = decide_special_category
    elif 'impute-by-mode' in how:
        get_impute_value = find_column_mode
    elif 'impute-by-mean' in how:
        get_impute_value = find_column_mean
    elif 'impute-by-median' in how:
        get_impute_value = find_column_median

    vals = {}
    for col in column_names:
        filtered_df = data[~data[col].isnull()][[col]].copy(deep=True)
        vals[col] = get_impute_value(filtered_df[col].values)

    print("Impute values: ", vals)
    data.fillna(value=vals, inplace=True)

    if how != 'drop-column':
        data[column_names] = data[column_names].round()
    return data


def initially_handle_nulls(X_data, missing):
    handle_nulls = {
        'special': missing,
    }
    # Checking dataset shape before handling nulls
    print("Dataset shape before handling nulls: ", X_data.shape)

    for how_to in handle_nulls.keys():
        X_data = handle_df_nulls(X_data, how_to, handle_nulls[how_to])
    # Checking dataset shape after handling nulls
    print("Dataset shape after handling nulls: ", X_data.shape)
    return X_data


def decide_special_category(data):
    """
    Description: Decides which value to designate as a special value, based on the values already in the data
    """
    if 0 not in data:
        return 0
    else:
        return max(data) + 1


def find_column_mode(data):
    result = stats.mode(data)
    return result.mode[0]


def find_column_mean(data):
    return np.mean(data).round()


def find_column_median(data):
    return np.median(data).round()
