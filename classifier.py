import pandas as pd
import random
from vectors import vector
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings




# def split_test_and_train(test_size, x_values, y_values):
#     combined = list(zip(x_values, y_values))
#     random.shuffle(combined)
#     x_values_shuffled, y_values_shuffled = zip(*combined)
#     split_index = int(len(x_values_shuffled) * test_size)
#     x_test = x_values_shuffled[:split_index]
#     y_test = y_values_shuffled[:split_index]
#     x_train = x_values_shuffled[split_index:]
#     y_train = y_values_shuffled[split_index:]
#     return list(x_test), list(y_test), list(x_train), list(y_train)


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

    # Suppress specific FutureWarning related to multi_class
    warnings.filterwarnings('ignore', category=FutureWarning, message=".*multi_class.*")
    # Model initialization
    # model = LogisticRegression(max_iter=200, multi_class='multinomial')
    model = RandomForestClassifier(n_estimators=100, random_state=42)

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
    vectors = vector()
    results = {}
    tag_list = get_tag_list()
    for i in range(1, 7):
        function_name = f"make_vector{i}"
        make_vector_function = getattr(vectors, function_name)
        vectors_i = make_vector_function()
        accuracy_i = perform_kfold_cv(vectors_i, tag_list)
        results[function_name] = accuracy_i
        print("Vector type {} avg accuracy {}".format(i, accuracy_i))
    # return results

def compare_vectors(vectors):
    # Generate vectors using make_vector2 and make_vector4
    vector2 = vectors.make_vector1()
    # print(vector2)
    vector4 = vectors.make_vector3()
    # print("$$$$$$")
    # print(vector4)


    # Check if the lengths of the vectors are the same
    if len(vector2) != len(vector4):
        print(f"Vectors have different lengths: {len(vector2)} vs {len(vector4)}")
        return

    # Compare each corresponding element in the two vectors
    differences = []
    for i in range(len(vector2)):
        diff = np.array(vector2[i]) - np.array(vector4[i])
        differences.append(diff)

        # Print differences for the first few elements for brevity
        if i < 5:
            print(f"Difference at index {i}: {diff}")

    # Calculate the overall difference
    total_difference = np.sum(np.abs(differences))
    mean_difference = np.mean(np.abs(differences))

    print(f"\nTotal difference between vectors: {total_difference}")
    print(f"Mean difference per element: {mean_difference}")

# Create an instance of the vector class
vectors = vector()
compare_vectors(vectors)
# calculate_accuracy_for_all_vectors()