from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.preprocessing import AgeImputer


numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
]

binary_features = [
    "AgeWasMissing",
    "CabinWasMissing",
    "IsAlone",
]


categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])


def make_preprocessor(scale=True):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipe = Pipeline(numeric_steps)

    column_transformer = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
        ("binary", "passthrough", binary_features),
    ])

    return Pipeline([
        ("age_imputer", AgeImputer()),
        ("columns", column_transformer),
    ])