"""
Hyperparameter Optimization for AutoML

Implements automated hyperparameter tuning using various optimization
strategies including grid search, random search, and Bayesian optimization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    optimization_method: str = "bayesian"  # grid, random, bayesian
    n_trials: int = 100
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    timeout_seconds: int = 3600
    n_jobs: int = -1
    random_state: int = 42


class HyperparameterOptimizer:
    """Hyperparameter optimization using various strategies."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_params = None
        self.best_score = None
        self.optimization_history = []

    def optimize_classification(
        self, X, y, model_class, param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for classification models."""
        logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")

        if self.config.optimization_method == "bayesian":
            return self._bayesian_optimization(X, y, model_class, param_space)
        elif self.config.optimization_method == "random":
            return self._random_search(X, y, model_class, param_space)
        elif self.config.optimization_method == "grid":
            return self._grid_search(X, y, model_class, param_space)
        else:
            raise ValueError(
                f"Unknown optimization method: {self.config.optimization_method}"
            )

    def _bayesian_optimization(
        self, X, y, model_class, param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, dict):
                    if param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, param_config["low"], param_config["high"]
                        )
                    elif param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"]
                        )
                else:
                    # Simple parameter space
                    params[param_name] = param_config

            # Create and evaluate model
            try:
                model = model_class(**params)
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                )
                return cv_scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {e}")
                return 0.0

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        # Store optimization history
        self.optimization_history = [
            {
                "trial": trial.number,
                "params": trial.params,
                "value": trial.value,
                "datetime": datetime.utcnow().isoformat(),
            }
            for trial in study.trials
        ]

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "method": "bayesian",
        }

    def _random_search(
        self, X, y, model_class, param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Random search optimization."""
        best_score = -np.inf
        best_params = None
        history = []

        for trial in range(self.config.n_trials):
            # Sample random parameters
            params = self._sample_parameters(param_space)

            try:
                model = model_class(**params)
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                )
                score = cv_scores.mean()

                history.append(
                    {
                        "trial": trial,
                        "params": params,
                        "value": score,
                        "datetime": datetime.utcnow().isoformat(),
                    }
                )

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue

        self.best_params = best_params
        self.best_score = best_score
        self.optimization_history = history

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "method": "random",
        }

    def _grid_search(
        self, X, y, model_class, param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Grid search optimization."""
        from sklearn.model_selection import ParameterGrid

        # Convert param_space to grid format
        grid_params = {}
        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                if param_config["type"] == "categorical":
                    grid_params[param_name] = param_config["choices"]
                elif param_config["type"] == "int":
                    grid_params[param_name] = list(
                        range(param_config["low"], param_config["high"] + 1)
                    )
                elif param_config["type"] == "float":
                    # Create grid for float parameters
                    grid_params[param_name] = np.linspace(
                        param_config["low"], param_config["high"], num=5
                    ).tolist()
            else:
                grid_params[param_name] = [param_config]

        # Generate parameter grid
        param_grid = list(ParameterGrid(grid_params))

        best_score = -np.inf
        best_params = None
        history = []

        for i, params in enumerate(param_grid):
            try:
                model = model_class(**params)
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs,
                )
                score = cv_scores.mean()

                history.append(
                    {
                        "trial": i,
                        "params": params,
                        "value": score,
                        "datetime": datetime.utcnow().isoformat(),
                    }
                )

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Grid trial {i} failed: {e}")
                continue

        self.best_params = best_params
        self.best_score = best_score
        self.optimization_history = history

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "method": "grid",
        }

    def _sample_parameters(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from parameter space."""
        params = {}

        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                if param_config["type"] == "categorical":
                    params[param_name] = np.random.choice(param_config["choices"])
                elif param_config["type"] == "int":
                    params[param_name] = np.random.randint(
                        param_config["low"], param_config["high"] + 1
                    )
                elif param_config["type"] == "float":
                    params[param_name] = np.random.uniform(
                        param_config["low"], param_config["high"]
                    )
            else:
                params[param_name] = param_config

        return params

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}

        df = pd.DataFrame(self.optimization_history)

        report = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "total_trials": len(self.optimization_history),
            "score_statistics": {
                "mean": df["value"].mean(),
                "std": df["value"].std(),
                "min": df["value"].min(),
                "max": df["value"].max(),
            },
            "optimization_method": self.config.optimization_method,
            "config": {
                "n_trials": self.config.n_trials,
                "cv_folds": self.config.cv_folds,
                "scoring_metric": self.config.scoring_metric,
            },
        }

        return report


class ModelSelector:
    """Automated model selection for different problem types."""

    def __init__(self):
        self.classification_models = {
            "logistic_regression": {
                "class": "sklearn.linear_model.LogisticRegression",
                "default_params": {"random_state": 42},
                "param_space": {
                    "C": {"type": "float", "low": 0.01, "high": 100.0},
                    "penalty": {
                        "type": "categorical",
                        "choices": ["l1", "l2", "elasticnet"],
                    },
                    "solver": {
                        "type": "categorical",
                        "choices": ["liblinear", "lbfgs", "saga"],
                    },
                },
            },
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestClassifier",
                "default_params": {"random_state": 42},
                "param_space": {
                    "n_estimators": {"type": "int", "low": 10, "high": 200},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
                },
            },
            "svm": {
                "class": "sklearn.svm.SVC",
                "default_params": {"random_state": 42},
                "param_space": {
                    "C": {"type": "float", "low": 0.01, "high": 100.0},
                    "kernel": {
                        "type": "categorical",
                        "choices": ["linear", "rbf", "poly"],
                    },
                    "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
                },
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingClassifier",
                "default_params": {"random_state": 42},
                "param_space": {
                    "n_estimators": {"type": "int", "low": 10, "high": 200},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                },
            },
        }

        self.regression_models = {
            "linear_regression": {
                "class": "sklearn.linear_model.LinearRegression",
                "default_params": {},
                "param_space": {},
            },
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "default_params": {"random_state": 42},
                "param_space": {
                    "n_estimators": {"type": "int", "low": 10, "high": 200},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                },
            },
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingRegressor",
                "default_params": {"random_state": 42},
                "param_space": {
                    "n_estimators": {"type": "int", "low": 10, "high": 200},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                },
            },
        }

    def select_best_classification_model(
        self, X, y, config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Select best classification model."""
        logger.info("Starting automated model selection for classification")

        results = {}
        optimizer = HyperparameterOptimizer(config)

        for model_name, model_info in self.classification_models.items():
            logger.info(f"Evaluating {model_name}")

            try:
                # Import model class
                module_path, class_name = model_info["class"].rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                model_class = getattr(module, class_name)

                # Optimize hyperparameters
                optimization_result = optimizer.optimize_classification(
                    X, y, model_class, model_info["param_space"]
                )

                results[model_name] = {
                    "best_score": optimization_result["best_score"],
                    "best_params": optimization_result["best_params"],
                    "optimization_history": optimization_result["optimization_history"],
                }

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        # Find best model
        best_model = max(
            [
                (name, result)
                for name, result in results.items()
                if "best_score" in result
            ],
            key=lambda x: x[1]["best_score"],
        )

        return {
            "best_model": best_model[0],
            "best_score": best_model[1]["best_score"],
            "best_params": best_model[1]["best_params"],
            "all_results": results,
        }

    def select_best_regression_model(
        self, X, y, config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Select best regression model."""
        logger.info("Starting automated model selection for regression")

        results = {}
        optimizer = HyperparameterOptimizer(config)

        for model_name, model_info in self.regression_models.items():
            logger.info(f"Evaluating {model_name}")

            try:
                # Import model class
                module_path, class_name = model_info["class"].rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                model_class = getattr(module, class_name)

                # Optimize hyperparameters
                optimization_result = optimizer.optimize_classification(
                    X, y, model_class, model_info["param_space"]
                )

                results[model_name] = {
                    "best_score": optimization_result["best_score"],
                    "best_params": optimization_result["best_params"],
                    "optimization_history": optimization_result["optimization_history"],
                }

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        # Find best model
        best_model = max(
            [
                (name, result)
                for name, result in results.items()
                if "best_score" in result
            ],
            key=lambda x: x[1]["best_score"],
        )

        return {
            "best_model": best_model[0],
            "best_score": best_model[1]["best_score"],
            "best_params": best_model[1]["best_params"],
            "all_results": results,
        }


class AutoMLPipeline:
    """Complete AutoML pipeline."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model_selector = ModelSelector()
        self.optimizer = HyperparameterOptimizer(config)

    def run_automl_classification(self, X, y) -> Dict[str, Any]:
        """Run complete AutoML pipeline for classification."""
        logger.info("Starting AutoML pipeline for classification")

        # Step 1: Model selection
        model_selection_result = self.model_selector.select_best_classification_model(
            X, y, self.config
        )

        # Step 2: Final optimization with best model
        best_model_name = model_selection_result["best_model"]
        best_model_info = self.model_selector.classification_models[best_model_name]

        # Import and optimize best model
        module_path, class_name = best_model_info["class"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)

        final_optimization = self.optimizer.optimize_classification(
            X, y, model_class, best_model_info["param_space"]
        )

        return {
            "model_selection": model_selection_result,
            "final_optimization": final_optimization,
            "recommended_model": {
                "name": best_model_name,
                "class": best_model_info["class"],
                "best_params": final_optimization["best_params"],
                "best_score": final_optimization["best_score"],
            },
        }

    def run_automl_regression(self, X, y) -> Dict[str, Any]:
        """Run complete AutoML pipeline for regression."""
        logger.info("Starting AutoML pipeline for regression")

        # Step 1: Model selection
        model_selection_result = self.model_selector.select_best_regression_model(
            X, y, self.config
        )

        # Step 2: Final optimization with best model
        best_model_name = model_selection_result["best_model"]
        best_model_info = self.model_selector.regression_models[best_model_name]

        # Import and optimize best model
        module_path, class_name = best_model_info["class"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)

        final_optimization = self.optimizer.optimize_classification(
            X, y, model_class, best_model_info["param_space"]
        )

        return {
            "model_selection": model_selection_result,
            "final_optimization": final_optimization,
            "recommended_model": {
                "name": best_model_name,
                "class": best_model_info["class"],
                "best_params": final_optimization["best_params"],
                "best_score": final_optimization["best_score"],
            },
        }
