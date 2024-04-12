from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
import xgboost as xgb


class MaterialsModels:
    """A class for training and evaluating materials models.

    This class provides a template for training and evaluating machine learning
    models on materials data. It includes methods for training models with
    cross-validation and randomized search for hyperparameter tuning, as well
    as evaluating the trained models.

    Args:
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_test (np.ndarray): The testing features.
        y_test (np.ndarray): The testing labels.
        save (bool, optional): Whether to save the trained model and metrics.
        Defaults to True.

    Attributes:
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_test (np.ndarray): The testing features.
        y_test (np.ndarray): The testing labels.
        save (bool): Whether to save the trained model and metrics.
        metrics (dict): The evaluation metrics.
        model: The trained model.
        _dir_out (Path): The output directory for saving the model and metrics.

    Raises:
        NotImplementedError: If the train_models method is not implemented.

    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        save: bool = True,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.save = save

        self.metrics = None
        self.model = None

        self._dir_out = Path("./models")

    def train_models(
        self, param_grid: dict = None, cv: int = 10, seed: int = 42
    ) -> dict:
        """Train XGBoost models with cross-validation and grid search.

        Args:
            param_grid (dict): The parameter grid for the grid search.
            Defaults to None.
            cv (int, optional): The number of cross-validation folds.
            Defaults to 5.
            seed (int, optional): The random seed for the train-test split.
            Defaults to 42.

        Returns:
            dict: The trained models.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Method not implemented")

    def evaluate_model(self) -> dict:
        """Evaluate the trained model.

        Returns:
            dict: The evaluation metrics.
        """
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        aucp = auc(self.y_test, y_pred)

        self.metrics = {
            "accuracy": accuracy,
            "confusion_matrix": confusion,
            "mean_squared_error": mse,
            "root_mean_squared_error": np.sqrt(mse),
            "auc": aucp,
        }
        return self.metrics


class XGBoostModels(MaterialsModels):
    """
    This class represents a set of XGBoost models for materials data.

    It inherits from the MaterialsModels base class and implements the
    train_models method specifically for XGBoost models. It uses
    cross-validation and grid search for training.

    Attributes:
        model: The trained model.
        metrics: The evaluation metrics for the model.
        x_train: The training data.
        y_train: The training labels.
        x_test: The test data.
        y_test: The test labels.
    """

    def train_models(
        self, param_grid: dict = None, cv: int = 10, seed: int = 42
    ) -> xgb.XGBRegressor:
        """Train XGBoost models with cross-validation and grid search.

        Args:
            param_grid (dict): The parameter grid for the grid search.
            Defaults to None.
            cv (int, optional): The number of cross-validation folds.
            Defaults to 5.
            seed (int, optional): The random seed for the train-test split.
            Defaults to 42.

        Returns:
            xgb.XGBRegressor: The best trained XGBoost model.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": randint(100, 1000),
                "max_depth": randint(3, 10),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5),
            }

        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.1,
            colsample_bytree=0.8,
            subsample=0.8,
            n_jobs=1,
        )
        kfold = KFold(n_splits=cv, random_state=seed, shuffle=True)
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=10,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            cv=kfold.split(self.x_train, self.y_train),
            verbose=3,
            random_state=seed,
        )
        random_search.fit(self.x_train, self.y_train)
        best_model = random_search.best_estimator_
        self.model = best_model

        params = random_search.best_params_
        print(f"Best Model:\n{best_model}")

        # train the best model on the full training set
        best_model.fit(self.x_train, self.y_train)
        y_pred = best_model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        if self.save:
            self._dir_out.mkdir(exist_ok=True)
            best_model.save_model(self._dir_out / "xgboost_model.json")
            pd.DataFrame(params, index=[0]).to_csv(
                self._dir_out / "xgboost_params.csv", index=False
            )

        return best_model

    def evaluate_model(self) -> dict:
        """Evaluate the trained model.

        Returns:
            dict: The evaluation metrics.
        """
        super().evaluate_model()

        if self.save:
            self._dir_out.mkdir(exist_ok=True)
            pd.DataFrame(self.metrics, index=[0]).to_csv(
                self._dir_out / "xgboost_metrics.csv", index=False
            )

        return self.metrics
