import pandas as pd
from vectors import vector
from initializer import DataStore
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings


def get_tag_list():
    file_path = 'Sentence_classification.xlsx'
    df = pd.read_excel(file_path)
    tag_list = []
    column_name = 'תיוג'
    if column_name in df.columns:
        tag_list = df[column_name].tolist()
    return tag_list


def perform_kfold_cv(X, y, model, print_results = False, file=None):
    # Initialize KFold with 10 splits
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Suppress specific FutureWarning related to multi_class
    warnings.filterwarnings('ignore', category=FutureWarning, message=".*multi_class.*")
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
        if print_results:
            file.write(str(y_pred)+ "\n")

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Calculate and return the mean accuracy across all folds
    mean_accuracy = np.mean(accuracies)
    # print(accuracies)
    return mean_accuracy


def calculate_accuracy_for_all_vectors(model, file):
    vectors = vector()
    results = {}
    tag_list = get_tag_list()
    best_accuracy = 0
    best_vector = 0
    for i in range(1, 30):
        function_name = f"make_vector{i}"
        make_vector_function = getattr(vectors, function_name)
        vectors_i = make_vector_function()
        accuracy_i = perform_kfold_cv(vectors_i, tag_list, model)
        results[function_name] = accuracy_i
        # Check if the current accuracy is the best so far
        if accuracy_i > best_accuracy:
            best_accuracy = accuracy_i
            best_vector = i

        print("Vector type {} avg accuracy {}".format(i, accuracy_i))
        file.write("Vector type {} avg accuracy {} \n".format(i, accuracy_i))
    return best_accuracy, best_vector

def run_on_different_nodels():
    logistic_regression_params = {'max_iter': 100, 'multi_class': 'multinomial'}
    random_forest_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}

    # Instantiate models with parameters
    models = [
        LogisticRegression(**logistic_regression_params),
        RandomForestClassifier(**random_forest_params),
    ]
    best_overall_accuracy = 0
    best_model = None
    best_vector_num = 0

    with open('results.txt', 'w', encoding='utf-8') as file:
        for model in models:
            file.write("model {} results \n".format(model.__class__.__name__))
            accuracy, best_vector = calculate_accuracy_for_all_vectors(model, file)

            # Check if this model's best accuracy is the overall best
            if accuracy > best_overall_accuracy:
                best_overall_accuracy = accuracy
                best_model = model
                best_vector_num = best_vector
    return best_model, best_vector_num, best_overall_accuracy


# print the classifications of the best vector
def print_best_vector(model, vector_num):
    function_name = f"make_vector{vector_num}"
    vectors = vector()
    make_vector_function = getattr(vectors, function_name)
    vectors_i = make_vector_function()
    tag_list = get_tag_list()
    with open('results_best_vector_classification.txt', 'w', encoding='utf-8') as file:
        perform_kfold_cv(vectors_i, tag_list, model,print_results=True, file=file)


# DataStore().analyze_excel_file()
best_model, best_vector_num, best_overall_accuracy = run_on_different_nodels()
print_best_vector(best_model, best_vector_num)
