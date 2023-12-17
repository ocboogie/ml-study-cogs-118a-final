import numpy as np
from sklearn import svm, ensemble, linear_model, neighbors, neural_network
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
)
from sklearn.compose import ColumnTransformer

from ucimlrepo import fetch_ucirepo

letter_set = fetch_ucirepo(id=59)

datasets = {
    "spam": {
        "dataset": fetch_ucirepo(id=94),
    },
    "iris": {
        "dataset": fetch_ucirepo(id=53),
        "output_preprocessor": lambda x: x == "Iris-setosa",
    },
    "BCW": {
        "dataset": fetch_ucirepo(id=17),
    },
    "mushroom": {
        "dataset": fetch_ucirepo(id=73),
        "preprocessor": ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    [
                        "cap-shape",
                        "cap-surface",
                        "bruises",
                        "odor",
                        "gill-attachment",
                        "gill-spacing",
                        "gill-size",
                        "gill-color",
                        "stalk-shape",
                        "stalk-root",
                        "stalk-surface-above-ring",
                        "stalk-surface-below-ring",
                        "stalk-color-above-ring",
                        "stalk-color-below-ring",
                        "veil-color",
                        "ring-number",
                        "ring-type",
                        "spore-print-color",
                        "population",
                        "habitat",
                    ],
                )
            ]
        ),
    },
    "letter.1": {"dataset": letter_set, "output_preprocessor": lambda x: x == "O"},
    "letter.2": {
        "dataset": letter_set,
        "output_preprocessor": lambda x: x < "N",
    },
}

algorithms = {
    "SVM": {
        "param_grid": {"C": [0.005, 0.1, 1, 2, 5, 10, 100]},
        "estimator": svm.LinearSVC(dual="auto", tol=1e-5, max_iter=10000),
    },
    "LOGREG": {
        "param_grid": lambda X: {
            "solver": ["liblinear" if X.shape[0] < 5000 else "saga"],
            "C": [0.0001, 0.01, 1, 100, 10000],
        },
        "estimator": linear_model.LogisticRegression(max_iter=10000),
        "preprocessor": RobustScaler(),
    },
    "RF": {
        "param_grid": {
            "n_estimators": [10, 100, 1000],
        },
        "estimator": ensemble.RandomForestClassifier(n_jobs=-1),
    },
    "KNN": {
        "param_grid": lambda X: {
            "n_neighbors": np.linspace(1, X.shape[0], 3).astype(int),
        },
        "estimator": neighbors.KNeighborsClassifier(weights="distance", n_jobs=-1),
    },
    "ANN": {
        "param_grid": lambda X: [
            {
                "solver": ["lbfgs" if X.shape[0] < 5000 else "adam"],
                "hidden_layer_sizes": [
                    (8,),
                    (128,),
                    (256,),
                ],
            },
        ],
        "estimator": neural_network.MLPClassifier(
            max_iter=10000,
            early_stopping=True,
        ),
        "preprocessor": RobustScaler(),
    },
}


def preprocess_dataset(dataset):
    X = dataset["dataset"].data.features
    y = dataset["dataset"].data.targets

    if "preprocessor" in dataset:
        preprocessor = dataset["preprocessor"]
        X = preprocessor.fit_transform(X)

    if "output_preprocessor" in dataset:
        output_preprocessor = dataset["output_preprocessor"]
        y = output_preprocessor(y)

    y = np.ravel(y.values)
    return [X, y]


for dataset_name in datasets:
    [X, y] = preprocess_dataset(datasets[dataset_name])

    for algorithm_name in algorithms:
        if "preprocessor" in algorithms[algorithm_name]:
            preprocessor = algorithms[algorithm_name]["preprocessor"]
            if "toarray" in dir(X):
                X = X.toarray()
            elif "values" in dir(X):
                X = X.values
            X = preprocessor.fit_transform(X)

        for test_split in [0.2, 0.5, 0.8]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=0
            )

            algorithm = algorithms[algorithm_name]
            estimator = algorithm["estimator"]
            param_grid = algorithm["param_grid"]

            if callable(param_grid):
                param_grid = param_grid(X_train)

            clf = GridSearchCV(
                estimator,
                param_grid,
                cv=5,
                scoring="accuracy",
                return_train_score=True,
                n_jobs=-1,
            )
            clf.fit(X, y)

            for mean_train_score, mean_test_score, params in zip(
                clf.cv_results_["mean_train_score"],
                clf.cv_results_["mean_test_score"],
                clf.cv_results_["params"],
            ):
                print(
                    "+%s,%s,%d/%d,%f,%f,%s"
                    # "Dataset: %s, Algorithm: %s, Testing split: %d/%d, Training Accuracy: %f, Validation Accuracy: %f, Params: %s"
                    % (
                        dataset_name,
                        algorithm_name,
                        (1 - test_split) * 100 + 0.1,
                        test_split * 100,
                        mean_train_score,
                        mean_test_score,
                        params,
                    )
                )

            train_score = clf.cv_results_["mean_train_score"][clf.best_index_]
            validation_score = clf.best_score_

            params = clf.best_params_

            model = algorithm["estimator"]
            model.set_params(**params)

            model.fit(X_train, y_train)

            test_score = model.score(X_test, y_test)

            print(
                # "Dataset: %s, Algorithm: %s, Testing split: %d/%d, Training Accuracy: %f, Validation Accuracy: %f, Test Accuracy: %f, Params: %s"
                "%s,%s,%d/%d,%f,%f,%f,%s"
                % (
                    dataset_name,
                    algorithm_name,
                    (1 - test_split) * 100 + 0.1,
                    test_split * 100,
                    train_score,
                    validation_score,
                    test_score,
                    params,
                )
            )
