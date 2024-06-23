# Exploring the Efficacy of Machine Learning Algorithms Across Datasets and Training Sizes

I wrote this paper as the final project for my COGS 118A (Introduction to Machine Learning) class. The assignment challenged me to implement and compare multiple machine learning algorithms across various datasets, following the framework established by [Caruana and Niculescu-Mizil's empirical study](https://doi.org/10.1145/1143844.1143865).

### Abstract

We test the efficacy of five machine learning algorithms (support vector machines,
logistic regression, random forests, K-nearest neighbors, artificial neural networks)
across diverse binary classification tasks using five UCI datasets. Employing a
search with K-fold cross-validation, we optimized hyperparameters and analyzed
model performance across three training/testing split ratios (20/80, 50/50, 80/20).
Random forests and artificial neural networks emerged as the top performing
models, with mean testing accuracies of 96.6%, while support vector machines
and logistic regression lagged behind with mean testing accuracies of 91.8% and
92.2%, respectively. With varying training/testing splits, we are able to analyze
how sensitivity models are to training set size, finding that random forests are the
most sensitive to different training/testing splits, whereas logistic regression is the
least sensitive. Finally, we analyze the optimal hyperparameters for each model and
split combination, and find that optimal hyperparameters are generally consistent
across training/testing splits, but vary across datasets.
