import numpy as np
import os 
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor



from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K Neighbours Regressor": KNeighborsRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Xgb Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "Adaboost Regressor": AdaBoostRegressor()
            }

            params = {
                "Linear Regression": {},  # No hyperparameters to tune

                "K Neighbours Regressor": {
                    "n_neighbors": [ 5, 7, 9,11]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error", "friedman_mse","poisson"]
                },

                "Random Forest Regressor": {
                    "n_estimators": [8,16,32,64,128,256]
                },

                "Gradient Boosting": {
                    "n_estimators":  [8,16,32,64,128,256],
                    "learning_rate": [0.01, 0.05, 0.1,0.001],
                    "subsample": [0.6,0.7,0.75,0.8,0.85,0.9, 1.0]
                },

                "Xgb Regressor": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [0.01, 0.05, 0.1,0.001]
                },

                "CatBoost Regressor": {
                    "depth": [6, 8,10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30,50,100]
                },

                "Adaboost Regressor": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [0.01, 0.05, 0.1,0.001]
                }
            }


            model_report, best_model = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise customException("Not a best score")
            
            logging.info("Best model found on both training and test data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise customException(e,sys)



    