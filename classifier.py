import pandas as pd
import random
from Vectors import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import numpy as np



def split_test_and_train(test_size, x_values, y_values):
    combined = list(zip(x_values, y_values))
    random.shuffle(combined)
    x_values_shuffled, y_values_shuffled = zip(*combined)
    split_index = int(len(x_values_shuffled) * test_size)
    x_test = x_values_shuffled[:split_index]
    y_test = y_values_shuffled[:split_index]
    x_train = x_values_shuffled[split_index:]
    y_train = y_values_shuffled[split_index:]
    return list(x_test), list(y_test), list(x_train), list(y_train)


def get_tag_list():
    file_path = 'Sentence_classification.xlsx'
    df = pd.read_excel(file_path)
    tag_list = []
    column_name = 'תיוג'
    if column_name in df.columns:
        tag_list = df[column_name].tolist()
    return tag_list



def perform_kfold_cv(X, y):
    # Initialize KFold with 10 splits
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Model initialization
    model = LogisticRegression(max_iter=200, multi_class='multinomial')

    # To store accuracy scores
    accuracies = []
    # Assuming X is currently a list or other data structure
    X = np.array(X)
    y = np.array(y)
    # Perform 10-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Calculate and return the mean accuracy across all folds
    mean_accuracy = np.mean(accuracies)
    print(accuracies)
    return mean_accuracy


def calculate_accuracy_for_all_vectors():
    results = {}
    for i in range(1, 4):
        function_name = f"call_to_vector{i}"
        make_vector_function = globals()[function_name]
        vectors_i = make_vector_function()
        tag_list = get_tag_list()
        accuracy_i = perform_kfold_cv(vectors_i, tag_list)
        results[function_name] = accuracy_i
        print("Vector type {} avg accuracy {}".format(i, accuracy_i))
    # return results



calculate_accuracy_for_all_vectors()