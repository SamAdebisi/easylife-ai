"""
Business Intelligence Dashboard

Provides advanced analytics and business intelligence capabilities for
monitoring ML model performance, business metrics, and operational insights.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


@dataclass
class BIConfig:
    """Configuration for Business Intelligence dashboard."""

    refresh_interval: int = 300  # seconds
    data_retention_days: int = 30
    alert_thresholds: Dict[str, float] = None
    dashboard_theme: str = "plotly_white"


class MetricsCollector:
    """Collects and aggregates business metrics."""

    def __init__(self, config: BIConfig):
        self.config = config
        self.metrics_data = []
        self.alert_thresholds = config.alert_thresholds or {}

    def collect_model_metrics(
        self, model_name: str, metrics: Dict[str, float], timestamp: datetime = None
    ):
        """Collect model performance metrics."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        metric_entry = {
            "timestamp": timestamp,
            "model_name": model_name,
            "metrics": metrics,
            "type": "model_performance",
        }

        self.metrics_data.append(metric_entry)
        logger.debug(f"Collected metrics for {model_name}: {metrics}")

    def collect_business_metrics(
        self, metrics: Dict[str, float], timestamp: datetime = None
    ):
        """Collect business metrics."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        metric_entry = {"timestamp": timestamp, "metrics": metrics, "type": "business"}

        self.metrics_data.append(metric_entry)
        logger.debug(f"Collected business metrics: {metrics}")

    def collect_operational_metrics(
        self, service_name: str, metrics: Dict[str, float], timestamp: datetime = None
    ):
        """Collect operational metrics."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        metric_entry = {
            "timestamp": timestamp,
            "service_name": service_name,
            "metrics": metrics,
            "type": "operational",
        }

        self.metrics_data.append(metric_entry)
        logger.debug(f"Collected operational metrics for {service_name}: {metrics}")

    def get_metrics_dataframe(
        self, metric_type: str = None, hours: int = 24
    ) -> pd.DataFrame:
        """Get metrics as DataFrame."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        filtered_data = [
            entry for entry in self.metrics_data if entry["timestamp"] >= cutoff_time
        ]

        if metric_type:
            filtered_data = [
                entry for entry in filtered_data if entry.get("type") == metric_type
            ]

        if not filtered_data:
            return pd.DataFrame()

        # Flatten metrics for DataFrame
        rows = []
        for entry in filtered_data:
            row = {
                "timestamp": entry["timestamp"],
                "type": entry.get("type", "unknown"),
                "model_name": entry.get("model_name", ""),
                "service_name": entry.get("service_name", ""),
            }
            row.update(entry.get("metrics", {}))
            rows.append(row)

        return pd.DataFrame(rows)


class DashboardGenerator:
    """Generates business intelligence dashboards."""

    def __init__(self, metrics_collector: MetricsCollector, config: BIConfig):
        self.metrics_collector = metrics_collector
        self.config = config

    def create_model_performance_dashboard(self) -> go.Figure:
        """Create model performance dashboard."""
        df = self.metrics_collector.get_metrics_dataframe("model_performance", hours=24)

        if df.empty:
            return self._create_empty_dashboard("No model performance data available")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Accuracy Over Time",
                "Loss Over Time",
                "Throughput",
                "Latency",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Plot accuracy
        for model in df["model_name"].unique():
            model_df = df[df["model_name"] == model]
            if "accuracy" in model_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=model_df["timestamp"],
                        y=model_df["accuracy"],
                        name=f"{model} Accuracy",
                        line=dict(width=2),
                    ),
                    row=1,
                    col=1,
                )

        # Plot loss
        for model in df["model_name"].unique():
            model_df = df[df["model_name"] == model]
            if "loss" in model_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=model_df["timestamp"],
                        y=model_df["loss"],
                        name=f"{model} Loss",
                        line=dict(width=2),
                    ),
                    row=1,
                    col=2,
                )

        # Plot throughput
        for model in df["model_name"].unique():
            model_df = df[df["model_name"] == model]
            if "throughput" in model_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=model_df["timestamp"],
                        y=model_df["throughput"],
                        name=f"{model} Throughput",
                        line=dict(width=2),
                    ),
                    row=2,
                    col=1,
                )

        # Plot latency
        for model in df["model_name"].unique():
            model_df = df[df["model_name"] == model]
            if "latency" in model_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=model_df["timestamp"],
                        y=model_df["latency"],
                        name=f"{model} Latency",
                        line=dict(width=2),
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="Model Performance Dashboard",
            height=800,
            showlegend=True,
            template=self.config.dashboard_theme,
        )

        return fig

    def create_business_metrics_dashboard(self) -> go.Figure:
        """Create business metrics dashboard."""
        df = self.metrics_collector.get_metrics_dataframe("business", hours=24)

        if df.empty:
            return self._create_empty_dashboard("No business metrics data available")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Revenue", "User Engagement", "API Calls", "Error Rate"),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Plot revenue
        if "revenue" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["revenue"],
                    name="Revenue",
                    line=dict(width=2, color="green"),
                ),
                row=1,
                col=1,
            )

        # Plot user engagement
        if "active_users" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["active_users"],
                    name="Active Users",
                    line=dict(width=2, color="blue"),
                ),
                row=1,
                col=2,
            )

        # Plot API calls
        if "api_calls" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["api_calls"],
                    name="API Calls",
                    line=dict(width=2, color="orange"),
                ),
                row=2,
                col=1,
            )

        # Plot error rate
        if "error_rate" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["error_rate"],
                    name="Error Rate",
                    line=dict(width=2, color="red"),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Business Metrics Dashboard",
            height=800,
            showlegend=True,
            template=self.config.dashboard_theme,
        )

        return fig

    def create_operational_dashboard(self) -> go.Figure:
        """Create operational metrics dashboard."""
        df = self.metrics_collector.get_metrics_dataframe("operational", hours=24)

        if df.empty:
            return self._create_empty_dashboard("No operational metrics data available")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "CPU Usage",
                "Memory Usage",
                "Response Time",
                "Queue Length",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Plot CPU usage
        for service in df["service_name"].unique():
            service_df = df[df["service_name"] == service]
            if "cpu_usage" in service_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=service_df["timestamp"],
                        y=service_df["cpu_usage"],
                        name=f"{service} CPU",
                        line=dict(width=2),
                    ),
                    row=1,
                    col=1,
                )

        # Plot memory usage
        for service in df["service_name"].unique():
            service_df = df[df["service_name"] == service]
            if "memory_usage" in service_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=service_df["timestamp"],
                        y=service_df["memory_usage"],
                        name=f"{service} Memory",
                        line=dict(width=2),
                    ),
                    row=1,
                    col=2,
                )

        # Plot response time
        for service in df["service_name"].unique():
            service_df = df[df["service_name"] == service]
            if "response_time" in service_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=service_df["timestamp"],
                        y=service_df["response_time"],
                        name=f"{service} Response Time",
                        line=dict(width=2),
                    ),
                    row=2,
                    col=1,
                )

        # Plot queue length
        for service in df["service_name"].unique():
            service_df = df[df["service_name"] == service]
            if "queue_length" in service_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=service_df["timestamp"],
                        y=service_df["queue_length"],
                        name=f"{service} Queue",
                        line=dict(width=2),
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="Operational Metrics Dashboard",
            height=800,
            showlegend=True,
            template=self.config.dashboard_theme,
        )

        return fig

    def create_alert_dashboard(self) -> go.Figure:
        """Create alert and threshold monitoring dashboard."""
        df = self.metrics_collector.get_metrics_dataframe(hours=24)

        if df.empty:
            return self._create_empty_dashboard("No metrics data available")

        # Create alert indicators
        alerts = []
        for _, row in df.iterrows():
            for metric, value in row.items():
                if metric in self.config.alert_thresholds:
                    threshold = self.config.alert_thresholds[metric]
                    if value > threshold:
                        alerts.append(
                            {
                                "timestamp": row["timestamp"],
                                "metric": metric,
                                "value": value,
                                "threshold": threshold,
                                "severity": (
                                    "high" if value > threshold * 1.5 else "medium"
                                ),
                            }
                        )

        if not alerts:
            return self._create_empty_dashboard("No alerts detected")

        # Create alert timeline
        fig = go.Figure()

        for alert in alerts:
            color = "red" if alert["severity"] == "high" else "orange"
            fig.add_trace(
                go.Scatter(
                    x=[alert["timestamp"]],
                    y=[alert["value"]],
                    mode="markers",
                    marker=dict(size=15, color=color),
                    name=f"{alert['metric']} Alert",
                    text=f"Threshold: {alert['threshold']}",
                    hovertemplate=f"<b>{alert['metric']}</b><br>"
                    + f"Value: {alert['value']}<br>"
                    + f"Threshold: {alert['threshold']}<br>"
                    + f"Severity: {alert['severity']}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Alert Dashboard",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            height=600,
            template=self.config.dashboard_theme,
        )

        return fig

    def _create_empty_dashboard(self, message: str) -> go.Figure:
        """Create empty dashboard with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            template=self.config.dashboard_theme,
        )
        return fig


class BusinessIntelligenceManager:
    """Manages business intelligence operations."""

    def __init__(self, config: BIConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.dashboard_generator = DashboardGenerator(self.metrics_collector, config)

    def collect_sample_metrics(self):
        """Collect sample metrics for demonstration."""
        # Sample model performance metrics
        models = ["nlp-service", "cv-service", "ts-forecasting", "recsys-service"]
        for model in models:
            for i in range(10):
                timestamp = datetime.utcnow() - timedelta(hours=i)
                metrics = {
                    "accuracy": np.random.uniform(0.8, 0.95),
                    "loss": np.random.uniform(0.1, 0.5),
                    "throughput": np.random.uniform(100, 1000),
                    "latency": np.random.uniform(10, 100),
                }
                self.metrics_collector.collect_model_metrics(model, metrics, timestamp)

        # Sample business metrics
        for i in range(24):
            timestamp = datetime.utcnow() - timedelta(hours=i)
            metrics = {
                "revenue": np.random.uniform(1000, 5000),
                "active_users": np.random.uniform(100, 1000),
                "api_calls": np.random.uniform(10000, 100000),
                "error_rate": np.random.uniform(0.01, 0.05),
            }
            self.metrics_collector.collect_business_metrics(metrics, timestamp)

        # Sample operational metrics
        services = ["nlp-service", "cv-service", "ts-forecasting", "recsys-service"]
        for service in services:
            for i in range(24):
                timestamp = datetime.utcnow() - timedelta(hours=i)
                metrics = {
                    "cpu_usage": np.random.uniform(20, 80),
                    "memory_usage": np.random.uniform(30, 90),
                    "response_time": np.random.uniform(50, 200),
                    "queue_length": np.random.uniform(0, 100),
                }
                self.metrics_collector.collect_operational_metrics(
                    service, metrics, timestamp
                )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        return {
            "model_performance": (
                self.dashboard_generator.create_model_performance_dashboard()
            ),
            "business_metrics": (
                self.dashboard_generator.create_business_metrics_dashboard()
            ),
            "operational": self.dashboard_generator.create_operational_dashboard(),
            "alerts": self.dashboard_generator.create_alert_dashboard(),
        }

    def export_dashboard_data(self, filepath: str):
        """Export dashboard data to file."""
        dashboard_data = self.get_dashboard_data()

        # Convert plots to JSON
        export_data = {}
        for name, fig in dashboard_data.items():
            export_data[name] = fig.to_json()

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Dashboard data exported to {filepath}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        df = self.metrics_collector.get_metrics_dataframe(hours=24)

        if df.empty:
            return {"message": "No data available"}

        summary = {
            "total_metrics": len(df),
            "models_tracked": (
                df["model_name"].nunique() if "model_name" in df.columns else 0
            ),
            "services_tracked": (
                df["service_name"].nunique() if "service_name" in df.columns else 0
            ),
            "time_range": {
                "start": df["timestamp"].min().isoformat() if not df.empty else None,
                "end": df["timestamp"].max().isoformat() if not df.empty else None,
            },
        }

        return summary
