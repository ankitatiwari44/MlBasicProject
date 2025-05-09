import sys
import os
import dill


import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import customException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path  = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_model = None
        best_score = -np.inf

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            if test_model_score > best_score:
                best_score = test_model_score
                best_model = model

        return report, best_model  # returning a tuple
    except Exception as e:
        raise customException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise customException(e,sys)
    