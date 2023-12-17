import pandas as pd

df = pd.read_csv(
    "model_performance.csv",
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
        "test_accuracy",
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


def num_format(x):
    if x < 0.999:
        return "{0:.3f}".format(x)[1:]
    else:
        return "1.0"


dataset_row = ""
line_row = ""
accuracy_row = ""
data_rows = ""

for model in df["model"].unique():
    for split in df["split"].unique():
        mean_train_accuracy = df[(df["model"] == model) & (df["split"] == split)][
            "train_accuracy"
        ].mean()

        mean_validation_accuracy = df[(df["model"] == model) & (df["split"] == split)][
            "validation_accuracy"
        ].mean()

        mean_test_accuracy = df[(df["model"] == model) & (df["split"] == split)][
            "test_accuracy"
        ].mean()

        df.loc[len(df.index)] = [
            "mean",
            model,
            split,
            mean_train_accuracy,
            mean_validation_accuracy,
            mean_test_accuracy,
            "",
        ]

for dataset in df["dataset"].unique():
    dataset_row += "& \\multicolumn{3}{c|}{" + rename[dataset] + "}"

column_number = 2
for dataset in df["dataset"].unique():
    line_row += "\\cmidrule(lr){{{0}-{1}}}".format(column_number, column_number + 2)
    column_number += 3

    accuracy_row += (
        "& {\\small\\textbf{MA}} & {\\small\\textbf{VA}} & {\\small\\textbf{TA}}"
    )

for model in df["model"].unique():
    for split in df["split"].unique():
        data_rows += "{0} ({1})".format(model, split)

        for dataset in df["dataset"].unique():
            row = df[
                (df["dataset"] == dataset)
                & (df["model"] == model)
                & (df["split"] == split)
            ]
            row = row.iloc[0]

            elpsilon = 0.001

            if (
                df[df["dataset"] == dataset]["train_accuracy"].max()
                - row["train_accuracy"]
                < elpsilon
                # and df[df["dataset"] == dataset]["train_accuracy"].max() < 0.99
            ):
                data_rows += " & \\textbf{{{0}}}".format(
                    num_format(row["train_accuracy"])
                )
            else:
                data_rows += " & {0}".format(num_format(row["train_accuracy"]))

            if (
                df[df["dataset"] == dataset]["validation_accuracy"].max()
                - row["validation_accuracy"]
                < elpsilon
                # and df[df["dataset"] == dataset]["validation_accuracy"].max() < 0.99
            ):
                data_rows += " & \\textbf{{{0}}}".format(
                    num_format(row["validation_accuracy"])
                )
            else:
                data_rows += " & {0}".format(num_format(row["validation_accuracy"]))

            if (
                df[df["dataset"] == dataset]["test_accuracy"].max()
                - row["test_accuracy"]
                < elpsilon
                # and df[df["dataset"] == dataset]["test_accuracy"].max() < 0.99
            ):
                data_rows += " & \\textbf{{{0}}}".format(
                    num_format(row["test_accuracy"])
                )
            else:
                data_rows += " & {0}".format(num_format(row["test_accuracy"]))

        data_rows += "\\\\ \n"
    data_rows += "\\midrule "

print(dataset_row + "\\\\")
print(line_row)
print("\\textbf{{Model (Split)}}{0} \\\\ \\midrule".format(accuracy_row))
print(data_rows)
