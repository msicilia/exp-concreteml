
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from itertools import product
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from multiprocessing import Pool
import yaml
from itertools import repeat
import importlib

def instantiate_models(model_config: dict, param_config: dict, fhe_config: dict):
    """Creates instances of the models with the given configuration"""
    #print(f"model_config: {model_config}")
    #print(f"param_config: {param_config}")
    #print(f"fhe_config: {fhe_config}")
    model_name, model_module = model_config["name"], model_config["module_name"]
    module = importlib.import_module(model_module)
    class_ = getattr(module, model_name)
    instance = class_(**param_config)
    fhe_model_name, fhe_model_module = model_config["fhe_name"], model_config["fhe_module_name"]
    module = importlib.import_module(fhe_model_module)
    class_ = getattr(module, fhe_model_name)
    instance_fhe = class_(**param_config, **fhe_config)
    return instance, instance_fhe
 

def expand_config(config: dict):
    """Expand a configuration dictionary into an interable"""
    match config["type"]:
        case "int":
            return range(config["min"], config["max"], config["step"])
        case _:
            raise ValueError(f"Unsupported type: {config['type']}")

def generate_concreteml_configs(concreteml_config: dict):
    return [(config_elem["param"]["name"], expand_config(config_elem["param"])) 
                for config_elem in concreteml_config["model_params"]]

def generate_model_configs(model_config: dict):
    return [(config_elem["param"]["name"], expand_config(config_elem["param"])) 
                for config_elem in model_config["params"]]

def generate_task_configs(task_config: dict):
    return [(config_elem["param"]["name"], expand_config(config_elem["param"])) 
                for config_elem in task_config["data"]["params"]]




def experiment(task_config: dict, concreteml_config: dict,  model_config: dict ):
    """"""
    # Generate configurations as lists of pairs (name, value iterators)
    task_configs = generate_task_configs(task_config)
    concreteml_model_configs = generate_concreteml_configs(concreteml_config)
    model_configs = generate_model_configs(model_config)
    task_config_names = [elem[0] for elem in task_configs]
    concreteml_model_config_names = [elem[0] for elem in concreteml_model_configs]
    model_config_names = [elem[0] for elem in model_configs]
    #print(f"task_configs: {task_config}")
    #print(f"concreteml_configs: {concreteml_model_configs}")
    #print(f"model_configs: {model_configs}")

    #stats = pd.DataFrame(columns=["model", "max_depth", "n_bits", "n_features", "training_time", "compilation_time", "prediction_time", "accuracy"])    
    names =  [elem[0] for elem in concreteml_model_configs + task_configs + model_configs]
    values = [elem[1] for elem in concreteml_model_configs + task_configs + model_configs]
    for vals in product(*values):
        named_values = dict(list(zip(names, vals)))
        #print(named_values)
        model, fhe_model = instantiate_models(model_config = model_config,
                                              fhe_config = { k:v for k, v in named_values.items() if k in concreteml_model_config_names},
                                              param_config = { k:v for k, v in named_values.items() if k in model_config_names})
        #print(model)
        #print(fhe_model)
        dataset_config = { k:v for k, v in named_values.items() if k in task_config_names}
        X, y = make_classification(random_state=42, n_samples=1_000, **dataset_config)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        results = {}

        tic = time.perf_counter()
        model.fit(X,y)
        toc = time.perf_counter()
        results["train_time"] = toc - tic
        #print(f"Training time: {train_time}")

        tic = time.perf_counter()
        fhe_model.fit(X,y)
        toc = time.perf_counter()
        results["train_time_fhe"] = toc - tic
        #print(f"Training time FHE: {train_time}")

        tic = time.perf_counter()
        fhe_model.compile(X_train)
        toc = time.perf_counter()
        results["compilation_time"] = toc - tic
        #print(f"Compilation time FHE: {compilation_time}")

        tic = time.perf_counter()
        y_pred = model.predict(X_test)
        toc = time.perf_counter()
        results["prediction_time"] = toc - tic
        #print(f"Prediction time: {prediction_time}")
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results["accuracy"] = acc
        results["f1"] = f1


        tic = time.perf_counter()
        y_pred = fhe_model.predict(X_test, fhe="execute")
        toc = time.perf_counter()
        results["prediction_time_fhe"] = toc - tic
        #print(f"Prediction time FHE: {prediction_time_fhe}")
        accuracy_fhe = accuracy_score(y_test, y_pred)
        f1_fhe = f1_score(y_test, y_pred)
        results["accuracy_fhe"] = accuracy_fhe
        results["f1_fhe"] = f1_fhe

        # print(f"model name: {model}, max_depth: {max_depth}, n_bits: {n_bits}, n_features: {n_features}")
        # acc = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        # print(f"Accuracy: {acc}, F1: {f1}")
        results.update(named_values)
        #new_row = {"model": modelname, "max_depth": max_depth, "n_bits": n_bits, 
        #             "n_features": n_features, 
        #             "training_time": train_time, 
        #         "compilation_time": compilation_time, 
        #         "prediction_time": prediction_time, 
        #         "accuracy": acc, "f1": f1}
        print(results)
        # stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
        #stats.to_csv(f"stats_{modelname}.csv")


if __name__ == '__main__':
    config = yaml.safe_load(open("config.yml"))
    n_models = len(config["models"])
    model_configs = [ m["model"] for m in config["models"]]
    configs = zip(repeat(config["task"]), repeat(config["concreteml"]), model_configs)
    with Pool(n_models) as p:
        p.starmap(experiment, configs)
