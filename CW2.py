# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Dawid Stepien
# STUDENT EMAIL: sc22ds@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    

def load_data(file_path, delimiter=','):
    num_rows, data, header_list=None, None, None
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None

    with open(file_path, 'r') as file:
        # open the file using pandas read_csv 
        datapd = pd.read_csv(file, delimiter=delimiter)
        # get the headers using pandas columns method
        header_list = datapd.columns
        # convert the data to numpy array
        data = np.array(datapd)
        # get the number of rows using the shape method
        num_rows = data.shape[0]
        
    return num_rows, data, header_list


def filter_data(data):
    # remove rows that contain -99
    return np.array([row for row in data if not -99 in row])


def statistics_data(data):
    coefficient_of_variation=[]
    data=filter_data(data)
    
    rotated_data = np.rot90(data, k=3)
    for feature in rotated_data:
        mean = np.mean(feature)
        std_dev = np.std(feature)
        if std_dev == 0:
            coefficient_of_variation.append(0)
        else:
            coefficient_of_variation.append(np.float64(std_dev/mean))

    return coefficient_of_variation


def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test=None, None, None, None
    np.random.seed(1)
    x = data[:, :-1]
    y = data[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    return x_train, x_test, y_train, y_test


def train_decision_tree(x_train, y_train,ccp_alpha=0):
    model=DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)
    return model


def make_predictions(model, X_test):    
    return model.predict(X_test)


def evaluate_model(model, x, y):    
    return np.float64(accuracy_score(y, model.predict(x))), np.float64(recall_score(y, model.predict(x)))


def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    ccp_alpha = 0

    model = train_decision_tree(x_train, y_train, ccp_alpha=0)
    original_accuracy, _ = evaluate_model(model, x_test, y_test)
    
    while True:
        
        model = train_decision_tree(x_train, y_train, ccp_alpha=ccp_alpha)
        acc_test, _ = evaluate_model(model, x_test, y_test)
        if  acc_test < original_accuracy - 0.01:
            break
        ccp_alpha += 0.0001

    return ccp_alpha - 0.0001


def tree_depths(model):
    return model.get_depth()


def important_feature(x_train, y_train,header_list):
    ccp_alpha = 0.0 
    depth = 2  

    while depth > 1:
        model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        model.fit(x_train, y_train)
        
        depth = model.get_depth()

        if depth > 1:
            ccp_alpha += 0.001
            
    feature_index = np.argmax(model.feature_importances_)
    feature_name = header_list[feature_index]

    return feature_name


# Example usage (Template Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
        
# References: 
# Here please provide recognition to any source if you have used or got code snippets from
# Please tell the lines that are relavant to that reference.
# For example: 
# Line 80-87 is inspired by a code at https://stackoverflow.com/questions/48414212/how-to-calculate-accuracy-from-decision-trees
