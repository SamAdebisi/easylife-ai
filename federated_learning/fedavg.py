"""
Federated Averaging (FedAvg) Implementation

Implements the FedAvg algorithm for federated learning with differential privacy
and secure aggregation capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class FedAvgConfig:
    """Configuration for Federated Averaging."""

    num_rounds: int = 100
    num_clients: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    secure_aggregation: bool = True


class FedAvgServer:
    """Federated Averaging Server implementation."""

    def __init__(self, config: FedAvgConfig):
        self.config = config
        self.global_model = None
        self.client_models = {}
        self.round_history = []

    def initialize_global_model(self, model: torch.nn.Module):
        """Initialize the global model."""
        self.global_model = model
        logger.info("Global model initialized")

    def aggregate_models(
        self, client_updates: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates using FedAvg."""
        if not client_updates:
            return self.global_model.state_dict()

        # Calculate weighted average
        total_samples = sum(
            update.get("num_samples", 1) for update in client_updates.values()
        )
        aggregated_params = {}

        for param_name in self.global_model.state_dict().keys():
            weighted_sum = torch.zeros_like(self.global_model.state_dict()[param_name])

            for client_id, update in client_updates.items():
                weight = update.get("num_samples", 1) / total_samples
                if param_name in update["model_state"]:
                    weighted_sum += weight * update["model_state"][param_name]
                else:
                    # Use global model parameters if client doesn't have this parameter
                    weighted_sum += weight * self.global_model.state_dict()[param_name]

            aggregated_params[param_name] = weighted_sum

        return aggregated_params

    def add_differential_privacy_noise(
        self, params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model parameters."""
        if not self.config.secure_aggregation:
            return params

        noisy_params = {}
        for param_name, param_tensor in params.items():
            # Calculate noise scale based on privacy budget
            noise_scale = (
                self.config.noise_multiplier
                * self.config.max_grad_norm
                / self.config.privacy_budget
            )
            noise = torch.normal(
                0, noise_scale, size=param_tensor.shape, device=param_tensor.device
            )
            noisy_params[param_name] = param_tensor + noise

        return noisy_params

    def update_global_model(self, client_updates: Dict[int, Dict[str, torch.Tensor]]):
        """Update global model with aggregated client updates."""
        # Aggregate client updates
        aggregated_params = self.aggregate_models(client_updates)

        # Add differential privacy noise
        if self.config.secure_aggregation:
            aggregated_params = self.add_differential_privacy_noise(aggregated_params)

        # Update global model
        self.global_model.load_state_dict(aggregated_params)

        # Log round statistics
        round_stats = {
            "round": len(self.round_history) + 1,
            "num_clients": len(client_updates),
            "total_samples": sum(
                update.get("num_samples", 1) for update in client_updates.values()
            ),
        }
        self.round_history.append(round_stats)

        logger.info(
            f"Round {round_stats['round']}: Updated global model with {round_stats['num_clients']} clients"
        )

    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state."""
        return self.global_model.state_dict()

    def evaluate_global_model(self, test_data) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_data:
                inputs, targets = batch
                outputs = self.global_model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_data)

        return {"accuracy": accuracy, "loss": avg_loss, "num_samples": total}


class FedAvgClient:
    """Federated Averaging Client implementation."""

    def __init__(
        self, client_id: int, local_data, model: torch.nn.Module, config: FedAvgConfig
    ):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.config = config
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.learning_rate
        )

    def local_training(
        self, global_model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform local training on client data."""
        # Load global model state
        self.model.load_state_dict(global_model_state)
        self.model.train()

        # Local training
        for epoch in range(self.config.local_epochs):
            for batch in self.local_data:
                inputs, targets = batch

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                loss.backward()

                # Gradient clipping for differential privacy
                if self.config.secure_aggregation:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()

        # Return model updates
        model_state = self.model.state_dict()
        num_samples = len(self.local_data.dataset)

        return {
            "model_state": model_state,
            "num_samples": num_samples,
            "client_id": self.client_id,
        }

    def evaluate_local_model(self, test_data) -> Dict[str, float]:
        """Evaluate local model on test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_data:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_data)

        return {"accuracy": accuracy, "loss": avg_loss, "num_samples": total}


class FederatedLearningOrchestrator:
    """Orchestrates federated learning process."""

    def __init__(self, config: FedAvgConfig):
        self.config = config
        self.server = FedAvgServer(config)
        self.clients = []

    def add_client(self, client: FedAvgClient):
        """Add a client to the federated learning process."""
        self.clients.append(client)
        logger.info(f"Added client {client.client_id}")

    def run_federated_training(self, test_data=None) -> List[Dict]:
        """Run federated learning training rounds."""
        logger.info(f"Starting federated learning with {len(self.clients)} clients")

        for round_num in range(self.config.num_rounds):
            logger.info(f"Starting round {round_num + 1}/{self.config.num_rounds}")

            # Select participating clients (random selection)
            participating_clients = np.random.choice(
                self.clients,
                size=min(self.config.num_clients, len(self.clients)),
                replace=False,
            )

            # Get global model state
            global_model_state = self.server.get_global_model_state()

            # Collect client updates
            client_updates = {}
            for client in participating_clients:
                try:
                    update = client.local_training(global_model_state)
                    client_updates[client.client_id] = update
                except Exception as e:
                    logger.error(f"Client {client.client_id} training failed: {e}")
                    continue

            # Update global model
            if client_updates:
                self.server.update_global_model(client_updates)

                # Evaluate global model if test data provided
                if test_data:
                    global_metrics = self.server.evaluate_global_model(test_data)
                    logger.info(
                        f"Round {round_num + 1} - Global Accuracy: {global_metrics['accuracy']:.4f}"
                    )

        return self.server.round_history
