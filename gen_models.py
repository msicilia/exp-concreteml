
MODELS_DIR = "./pickle_field"
from sklearn.metrics import accuracy_score
from concrete.ml.sklearn import DecisionTreeClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from itertools import product
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
import time
import pandas as pd


models = ["DecisionTreeClassifier"]# , "RandomForestClassifier"]
max_depths = range(3, 6)
n_bits = range(2, 4)
n_features = range(5, 8)
 
stats = pd.DataFrame(columns=["model", "max_depth", "n_bits", "n_features", "training_time", "compilation_time", "prediction_time", "accuracy"])    
for model, max_depth, n_bits, n_features in product(models, max_depths, n_bits, n_features):
    X, y = make_classification(random_state=42, n_features=n_features, n_samples=1_000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    if model == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
    tic = time.perf_counter()
    model.fit(X,y)
    toc = time.perf_counter()
    traint_time = toc - tic
    print(f"Training time: {train_time}")
    tic = time.perf_counter()
    model.compile(X_train)
    toc = time.perf_counter()
    compilation_time = toc - tic
    print(f"Compilation time: {compilation_time}")
    tic = time.perf_counter()
    y_pred = model.predict(X_test, fhe="execute")
    toc = time.perf_counter()
    prediction_time = toc - tic
    print(f"Prediction time: {prediction_time}")
    #with open(f"{MODELS_DIR}/{model}_{max_depth}_{n_bits}_{n_features}.pkl", "wb") as f:
    #    pickle.dump(model, f) 
    print(f"model name: {model}, max_depth: {max_depth}, n_bits: {n_bits}, n_features: {n_features}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    new_row = {"model": model, "max_depth": max_depth, "n_bits": n_bits, 
                "n_features": n_features, 
                "training_time": traint_time, 
               "compilation_time": compilation_time, 
               "prediction_time": prediction_time, 
               "accuracy": acc}
    stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
stats.to_csv("stats.csv")