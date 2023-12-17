import json
import pandas as pd

df = pd.read_csv(
    "hyperparameter_performance.csv",
    sep=",",
    escapechar="\\",
    quotechar='"',
    header=None,
    names=[
        "dataset",
        "model",
        "split",
        "train_accuracy",
        "validation_accuracy",
        "parameters",
    ],
)

rename = {
    "iris": "Iris",
    "BCW": "BCW",
    "mushroom": "Mushroom",
    "spam": "Spambase",
    "letter.1": "Letter.1",
    "letter.2": "Letter.2",
    "mean": "Mean",
}

fields = {
    "SVM": {"C": "C"},
    "LOGREG": {"C": "C"},
    "RF": {"n_estimators": "N"},
    "KNN": {"n_neighbors": "N"},
    "ANN": {"hidden_layer_sizes": "H"},
}

model_row = ""
line_row = ""
param_row = ""
data_rows = ""

column_number = 2
for model in df["model"].unique():
    model_row += "& \\multicolumn{1}{c|}{" + model + "}"
    line_row += "\\cmidrule(lr){{{0}-{1}}}".format(column_number, column_number + 1)
    column_number += 1

    for field in fields[model].values():
        param_row += "& \\multicolumn{1}{c|}{" + field + "}"

for dataset in df["dataset"].unique():
    for split in df["split"].unique():
        data_rows += "{0} ({1})".format(rename[dataset], split)

        for model in df["model"].unique():
            cases = df[
                (df["dataset"] == dataset)
                & (df["model"] == model)
                & (df["split"] == split)
            ]
            row = df.iloc[cases["validation_accuracy"].idxmax()]

            for field in fields[model]:
                data_rows += "& {0}".format(json.loads(row["parameters"])[field])

        data_rows += "\\\\ \n"
    data_rows += "\\midrule "

print(model_row + "\\\\")
print(line_row)
print("\\textbf{{Dataset (Split)}}{0} \\\\ \\midrule".format(param_row))
print(data_rows)
