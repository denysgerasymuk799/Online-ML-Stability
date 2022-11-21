from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


SEED = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = TEST_FRACTION
TRAIN_FRACTION = 1 - VALIDATION_FRACTION - TEST_FRACTION
CV_BOOTSTRAP_FRACTION = 0.6
TRAIN_CV_BOOTSTRAP_FRACTION = TRAIN_FRACTION + VALIDATION_FRACTION * CV_BOOTSTRAP_FRACTION

FOLKTABLES_COLUMN_TO_TYPE = {
    "categorical": ['SCHL', 'MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC'],
    "numerical": ['AGEP']
}

DATASETS_CONFIG = {
    'folktables': {
        'target_column': 'ESR',
        'numerical_features': ['AGEP', 'SCHL']
    },
    'phishing': {
        'target_column': 'is_phishing',
    }
}

TEST_GROUPS_CONFIG = {
    'folktables': {
        "Race": {
            "column_name": "RAC1P",
            "preprocess": 0,
            "advantaged": 1,
            "disadvantaged": 2
        },
        "Sex": {
            "column_name": "SEX",
            "preprocess": 0,
            "advantaged": 1,
            "disadvantaged": 2
        }
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
