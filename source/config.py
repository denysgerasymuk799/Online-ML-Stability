from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


SEED = 42
TEST_SIZE = 0.2

FOLKTABLES_COLUMN_TO_TYPE = {
    "categorical": ['SCHL', 'MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC'],
    "numerical": ['AGEP']
}

DATASETS_CONFIG = {
    'folktables': {
        'target_column': 'ESR',
        'numerical_features': ['AGEP', 'SCHL']
    }
}

MODELS_CONFIG = [
    {
        'model_name': 'XGBClassifier',
        'model': XGBClassifier(random_state=SEED, verbosity = 0),
        'params': {
            'learning_rate': [0.1],
            'n_estimators': [10, 100, 200],
            'max_depth': range(5, 16, 5),
            'objective':  ['binary:logistic'],
        }
    },
    {
        'model_name': 'RandomForestClassifier',
        'model': RandomForestClassifier(random_state=SEED),
        'params': {
            "max_depth": [3, 4, 6, 10],
            "min_samples_split": [2, 6],
            "min_samples_leaf": [1, 2, 4],
            "n_estimators": [10, 20, 50, 100],
        }
    }
]
