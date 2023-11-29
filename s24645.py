import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "car_evaluation.data",
    names=["a", "b", "c", "d", "e", "f", "class"],
)

features = df.drop("class", axis=1)
labels = df["class"]
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=10
)


def class_probability(y_train):
    class_probability = {}
    for class_label in y_train.unique():
        class_probability[class_label] = y_train[y_train == class_label].count() / len(
            y_train
        )
    return class_probability


def feature_prob(X, y, laplace=1):
    feature_prob = {label: {} for label in y.unique()}
    class_counts = y.value_counts()
    for feature in X.columns:
        feature_counts = X.groupby([y, feature]).size()
        feature_counts += laplace
        feature_probs = feature_counts.div(
            class_counts + laplace * len(X[feature].unique()), axis=0
        )
        for label in feature_prob:
            feature_prob[label][feature] = feature_probs.loc[label].to_dict()
    return feature_prob


def classification(X, class_probs, feature_probs):
    predictions = []
    for _, row in X.iterrows():
        log_probabilities = {}
        for class_label in class_probs:
            log_prob = np.log(class_probs[class_label])
            for feature in X.columns:
                feature_value = row[feature]
                feature_value_prob = feature_probs[class_label][feature].get(
                    feature_value, 0.00001
                )
                log_prob += np.log(feature_value_prob)
            log_probabilities[class_label] = log_prob
        predicted_class = max(log_probabilities, key=log_probabilities.get)
        predictions.append(predicted_class)
    return predictions


a = feature_prob(x_train, y_train)
b = class_probability(y_train)
predictions = classification(x_test, b, a)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


# Using KFold for cross-validation
# Uzycie cross-validation daje gorszy wynik niz proste podzielenie na train test 07 03

# kf = KFold(n_splits=5, random_state=10, shuffle=True)
# accuracies = []

# for train_index, test_index in kf.split(features):
#     x_train, x_test = features.iloc[train_index], features.iloc[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]

#     # Use your existing functions with the training and testing data
#     a = feature_prob(x_train, y_train)
#     b = class_probability(y_train)
#     predictions = classification(x_test, b, a)

#     # Calculate accuracy and append to list
#     accuracy = accuracy_score(y_test, predictions)
#     accuracies.append(accuracy)

# # Compute average accuracy across all folds
# average_accuracy = np.mean(accuracies)
# print(f"Average Accuracy: {average_accuracy}")

# Accuracy: 0.8689788053949904