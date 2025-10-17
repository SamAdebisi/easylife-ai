"""
Model Explainer for Explainable AI

Implements SHAP, LIME, and other interpretability techniques to provide
explanations for model predictions and build trust in AI systems.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Try to import explainability libraries
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    import lime.lime_text

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

logger = logging.getLogger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for model explanations."""

    method: str = "shap"  # shap, lime, both
    background_samples: int = 100
    max_features: int = 10
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None


class SHAPExplainer:
    """SHAP-based model explainer."""

    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.explainer = None
        self.shap_values = None

    def fit_explainer(self, model, X_train: np.ndarray, X_test: np.ndarray = None):
        """Fit SHAP explainer to model."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        logger.info("Fitting SHAP explainer")

        # Create background dataset
        if len(X_train) > self.config.background_samples:
            background_indices = np.random.choice(
                len(X_train), self.config.background_samples, replace=False
            )
            background = X_train[background_indices]
        else:
            background = X_train

        # Create explainer based on model type
        if hasattr(model, "predict_proba"):
            # Tree-based models
            self.explainer = shap.TreeExplainer(model)
        else:
            # Linear models or other
            self.explainer = shap.Explainer(model, background)

        # Calculate SHAP values
        if X_test is not None:
            self.shap_values = self.explainer.shap_values(X_test)
        else:
            self.shap_values = self.explainer.shap_values(background)

        logger.info("SHAP explainer fitted successfully")

    def explain_prediction(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain a single prediction."""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")

        # Get SHAP values for instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        # Get feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            shap_values = shap_values[0]  # Use first class

        feature_importance = np.abs(shap_values[0])
        feature_names = self.config.feature_names or [
            f"Feature_{i}" for i in range(len(feature_importance))
        ]

        # Get top features
        top_indices = np.argsort(feature_importance)[-self.config.max_features :][::-1]

        explanation = {
            "prediction": self.explainer.expected_value + np.sum(shap_values[0]),
            "base_value": self.explainer.expected_value,
            "feature_contributions": {
                feature_names[i]: {
                    "value": instance[i],
                    "shap_value": shap_values[0][i],
                    "importance": feature_importance[i],
                }
                for i in top_indices
            },
            "top_features": [
                {
                    "feature": feature_names[i],
                    "importance": feature_importance[i],
                    "contribution": shap_values[0][i],
                }
                for i in top_indices
            ],
        }

        return explanation

    def get_global_explanations(self) -> Dict[str, Any]:
        """Get global model explanations."""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit_explainer first.")

        # Calculate feature importance
        if isinstance(self.shap_values, list):
            shap_values = np.array(self.shap_values[0])
        else:
            shap_values = np.array(self.shap_values)

        feature_importance = np.mean(np.abs(shap_values), axis=0)
        feature_names = self.config.feature_names or [
            f"Feature_{i}" for i in range(len(feature_importance))
        ]

        # Get top features
        top_indices = np.argsort(feature_importance)[-self.config.max_features :][::-1]

        global_explanation = {
            "feature_importance": {
                feature_names[i]: feature_importance[i] for i in top_indices
            },
            "top_features": [
                {"feature": feature_names[i], "importance": feature_importance[i]}
                for i in top_indices
            ],
            "summary_stats": {
                "mean_importance": np.mean(feature_importance),
                "std_importance": np.std(feature_importance),
                "max_importance": np.max(feature_importance),
            },
        }

        return global_explanation


class LIMEExplainer:
    """LIME-based model explainer."""

    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.explainer = None

    def fit_explainer(
        self, model, X_train: np.ndarray, feature_names: List[str] = None
    ):
        """Fit LIME explainer to model."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")

        logger.info("Fitting LIME explainer")

        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names
            or [f"Feature_{i}" for i in range(X_train.shape[1])],
            class_names=self.config.class_names,
            mode="classification" if hasattr(model, "predict_proba") else "regression",
        )

        logger.info("LIME explainer fitted successfully")

    def explain_prediction(self, instance: np.ndarray, model) -> Dict[str, Any]:
        """Explain a single prediction using LIME."""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")

        # Get LIME explanation
        explanation = self.explainer.explain_instance(
            instance,
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            num_features=self.config.max_features,
        )

        # Extract explanation data
        explanation_data = {
            "prediction": model.predict(instance.reshape(1, -1))[0],
            "explanation": explanation.as_list(),
            "top_features": [
                {"feature": feature, "weight": weight}
                for feature, weight in explanation.as_list()
            ],
        }

        return explanation_data


class ModelExplainer:
    """Main model explainer combining multiple techniques."""

    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.shap_explainer = SHAPExplainer(config) if SHAP_AVAILABLE else None
        self.lime_explainer = LIMEExplainer(config) if LIME_AVAILABLE else None
        self.explanations = {}

    def fit_explainers(self, model, X_train: np.ndarray, X_test: np.ndarray = None):
        """Fit all available explainers."""
        logger.info("Fitting model explainers")

        if self.config.method in ["shap", "both"] and self.shap_explainer:
            self.shap_explainer.fit_explainer(model, X_train, X_test)

        if self.config.method in ["lime", "both"] and self.lime_explainer:
            self.lime_explainer.fit_explainer(model, X_train, self.config.feature_names)

        logger.info("Model explainers fitted successfully")

    def explain_prediction(self, instance: np.ndarray, model) -> Dict[str, Any]:
        """Explain a single prediction using available methods."""
        explanations = {}

        if self.config.method in ["shap", "both"] and self.shap_explainer:
            try:
                shap_explanation = self.shap_explainer.explain_prediction(instance)
                explanations["shap"] = shap_explanation
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
                explanations["shap"] = {"error": str(e)}

        if self.config.method in ["lime", "both"] and self.lime_explainer:
            try:
                lime_explanation = self.lime_explainer.explain_prediction(
                    instance, model
                )
                explanations["lime"] = lime_explanation
            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")
                explanations["lime"] = {"error": str(e)}

        # Store explanation
        explanation_id = f"explanation_{datetime.utcnow().timestamp()}"
        self.explanations[explanation_id] = {
            "instance": instance.tolist(),
            "explanations": explanations,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return explanations

    def get_global_explanations(self) -> Dict[str, Any]:
        """Get global model explanations."""
        global_explanations = {}

        if self.config.method in ["shap", "both"] and self.shap_explainer:
            try:
                shap_global = self.shap_explainer.get_global_explanations()
                global_explanations["shap"] = shap_global
            except Exception as e:
                logger.error(f"SHAP global explanation failed: {e}")
                global_explanations["shap"] = {"error": str(e)}

        return global_explanations

    def generate_explanation_report(
        self, model, X_test: np.ndarray, y_test: np.ndarray = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation report."""
        logger.info("Generating explanation report")

        report = {
            "model_info": {
                "type": type(model).__name__,
                "feature_count": X_test.shape[1],
                "sample_count": X_test.shape[0],
            },
            "explanation_config": {
                "method": self.config.method,
                "max_features": self.config.max_features,
                "background_samples": self.config.background_samples,
            },
            "global_explanations": self.get_global_explanations(),
            "sample_explanations": [],
        }

        # Generate explanations for sample instances
        sample_indices = np.random.choice(
            len(X_test), min(5, len(X_test)), replace=False
        )

        for idx in sample_indices:
            instance = X_test[idx]
            explanation = self.explain_prediction(instance, model)

            sample_explanation = {
                "instance_id": idx,
                "instance": instance.tolist(),
                "prediction": model.predict(instance.reshape(1, -1))[0],
                "explanations": explanation,
            }

            if y_test is not None:
                sample_explanation["actual_label"] = y_test[idx]

            report["sample_explanations"].append(sample_explanation)

        return report

    def save_explanations(self, filepath: str):
        """Save explanations to file."""
        with open(filepath, "w") as f:
            json.dump(self.explanations, f, indent=2)

        logger.info(f"Explanations saved to {filepath}")

    def load_explanations(self, filepath: str):
        """Load explanations from file."""
        with open(filepath, "r") as f:
            self.explanations = json.load(f)

        logger.info(f"Explanations loaded from {filepath}")


class ExplainableAIManager:
    """Manages explainable AI operations."""

    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.explainer = ModelExplainer(config)

    def explain_model(
        self, model, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray = None
    ) -> Dict[str, Any]:
        """Explain model predictions."""
        logger.info("Starting model explanation process")

        # Fit explainers
        self.explainer.fit_explainers(model, X_train, X_test)

        # Generate comprehensive report
        report = self.explainer.generate_explanation_report(model, X_test, y_test)

        logger.info("Model explanation completed")
        return report

    def get_feature_importance_ranking(
        self, model, X_train: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Get feature importance ranking."""
        if not self.explainer.shap_explainer:
            raise ValueError("SHAP explainer not available")

        # Fit SHAP explainer
        self.explainer.shap_explainer.fit_explainer(model, X_train)

        # Get global explanations
        global_explanations = self.explainer.shap_explainer.get_global_explanations()

        # Create ranking
        ranking = []
        for feature, importance in global_explanations["feature_importance"].items():
            ranking.append(
                {
                    "feature": feature,
                    "importance": importance,
                    "rank": 0,  # Will be set after sorting
                }
            )

        # Sort by importance and assign ranks
        ranking.sort(key=lambda x: x["importance"], reverse=True)
        for i, item in enumerate(ranking):
            item["rank"] = i + 1

        return ranking

    def compare_model_explanations(
        self, models: Dict[str, Any], X_test: np.ndarray
    ) -> Dict[str, Any]:
        """Compare explanations across multiple models."""
        logger.info("Comparing model explanations")

        comparison = {}

        for model_name, model in models.items():
            try:
                # Generate explanations for this model
                report = self.explain_model(model, X_test, X_test)
                comparison[model_name] = report
            except Exception as e:
                logger.error(f"Error explaining {model_name}: {e}")
                comparison[model_name] = {"error": str(e)}

        return comparison
