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

model_row = ""
line_row = ""
dataset_row = ""
data_rows = ""

for dataset in df["dataset"].unique():
    dataset_row += "& \\multicolumn{1}{c|}{" + rename[dataset] + "}"

dataset_row += "& \\multicolumn{1}{c}{Mean}"

for model in df["model"].unique():
    data_rows += model

    sum = 0

    for dataset in df["dataset"].unique():
        std = df[(df["dataset"] == dataset) & (df["model"] == model)][
            "test_accuracy"
        ].std()

        sum += std

        data_rows += "& {0:.3f}".format(std)

    data_rows += "& {0:.3f}".format(sum / len(df["dataset"].unique()))
    data_rows += "\\\\ \n"


print(line_row)
print("\\textbf{{Model}}{0} \\\\ \\midrule".format(dataset_row))
print(data_rows)
