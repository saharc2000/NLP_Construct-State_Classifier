import pandas as pd
from vectors import vector
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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


def perform_kfold_cv(X, y, model):
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
    for i in range(1, 31):
        function_name = f"make_vector{i}"
        make_vector_function = getattr(vectors, function_name)
        vectors_i = make_vector_function()
        accuracy_i = perform_kfold_cv(vectors_i, tag_list, model)
        results[function_name] = accuracy_i
        print("Vector type {} avg accuracy {}".format(i, accuracy_i))
        file.write("Vector type {} avg accuracy {} \n".format(i, accuracy_i))
    # return results

def run_on_different_nodels():
    logistic_regression_params = {'max_iter': 200, 'multi_class': 'multinomial'}
    random_forest_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    gradient_boosting_params = {'n_estimators': 50, 'learning_rate': 0.2, 'max_depth': 2, 'random_state': 42}

    # Instantiate models with parameters
    models = [
        LogisticRegression(**logistic_regression_params),
        RandomForestClassifier(**random_forest_params),
        GradientBoostingClassifier(**gradient_boosting_params),
    ]
    with open('results.txt', 'w', encoding='utf-8') as file:
        for model in models:
            file.write("model {} results \n".format(model.__class__.__name__))
            calculate_accuracy_for_all_vectors(model, file)


# models = [LogisticRegression(), RandomForestClassifier(), XGBClassifier()]
# calculate_accuracy_for_all_vectors(models)
# calculate_accuracy_for_all_vectors()
run_on_different_nodels()