"""
Model Compression Techniques

Implements various model compression techniques including quantization,
pruning, knowledge distillation, and neural architecture search.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for model compression."""

    target_compression_ratio: float = 0.5  # Target compression ratio
    quantization_bits: int = 8
    pruning_ratio: float = 0.3
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_distillation: bool = True


class ModelQuantizer:
    """Advanced model quantization techniques."""

    def __init__(self, config: CompressionConfig):
        self.config = config

    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        logger.info("Applying dynamic quantization")

        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )

        logger.info("Dynamic quantization completed")
        return quantized_model

    def static_quantization(
        self, model: nn.Module, calibration_data: torch.Tensor
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        logger.info("Applying static quantization")

        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # Calibrate with sample data
        model_prepared.eval()
        with torch.no_grad():
            for i in range(min(100, len(calibration_data))):
                sample = calibration_data[i : i + 1]
                _ = model_prepared(sample)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)

        logger.info("Static quantization completed")
        return quantized_model

    def post_training_quantization(
        self, model: nn.Module, calibration_data: torch.Tensor
    ) -> nn.Module:
        """Apply post-training quantization."""
        logger.info("Applying post-training quantization")

        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # Prepare model
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # Calibrate
        model_prepared.eval()
        with torch.no_grad():
            for i in range(min(100, len(calibration_data))):
                sample = calibration_data[i : i + 1]
                _ = model_prepared(sample)

        # Convert
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)

        logger.info("Post-training quantization completed")
        return quantized_model


class ModelPruner:
    """Advanced model pruning techniques."""

    def __init__(self, config: CompressionConfig):
        self.config = config

    def magnitude_pruning(self, model: nn.Module) -> nn.Module:
        """Apply magnitude-based pruning."""
        logger.info(
            f"Applying magnitude pruning with ratio {self.config.pruning_ratio}"
        )

        # Get parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        # Apply unstructured pruning
        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=self.config.pruning_ratio,
        )

        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            torch.nn.utils.prune.remove(module, param_name)

        logger.info("Magnitude pruning completed")
        return model

    def structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning."""
        logger.info("Applying structured pruning")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune entire neurons
                num_features = module.out_features
                num_to_prune = int(num_features * self.config.pruning_ratio)

                if num_to_prune > 0:
                    # Get magnitude of each neuron
                    neuron_magnitudes = torch.norm(module.weight, dim=1)
                    _, indices_to_prune = torch.topk(
                        neuron_magnitudes, num_to_prune, largest=False
                    )

                    # Zero out pruned neurons
                    module.weight.data[indices_to_prune] = 0
                    if module.bias is not None:
                        module.bias.data[indices_to_prune] = 0

        logger.info("Structured pruning completed")
        return model

    def lottery_ticket_hypothesis(
        self, model: nn.Module, train_loader, epochs: int = 10
    ) -> nn.Module:
        """Apply lottery ticket hypothesis pruning."""
        logger.info("Applying lottery ticket hypothesis pruning")

        # Save original weights
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()

        # Train for a few epochs
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Apply magnitude pruning
        pruned_model = self.magnitude_pruning(model)

        # Reset to original weights
        for name, param in pruned_model.named_parameters():
            if name in original_weights:
                param.data = original_weights[name]

        logger.info("Lottery ticket hypothesis pruning completed")
        return pruned_model


class KnowledgeDistillation:
    """Knowledge distillation for model compression."""

    def __init__(self, config: CompressionConfig):
        self.config = config

    def distill_knowledge(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader,
        epochs: int = 50,
    ) -> nn.Module:
        """Distill knowledge from teacher to student model."""
        logger.info("Starting knowledge distillation")

        teacher_model.eval()
        student_model.train()

        optimizer = torch.optim.Adam(student_model.parameters())
        criterion = nn.KLDivLoss(reduction="batchmean")
        # mse_loss = nn.MSELoss()  # Not used in this implementation

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                    teacher_probs = torch.softmax(
                        teacher_outputs / self.config.distillation_temperature, dim=1
                    )

                # Get student predictions
                student_outputs = student_model(data)
                student_probs = torch.log_softmax(
                    student_outputs / self.config.distillation_temperature, dim=1
                )

                # Calculate distillation loss
                distillation_loss = criterion(student_probs, teacher_probs)

                # Calculate student loss
                student_loss = nn.CrossEntropyLoss()(student_outputs, target)

                # Combined loss
                loss = (
                    self.config.distillation_alpha
                    * distillation_loss
                    * (self.config.distillation_temperature**2)
                    + (1 - self.config.distillation_alpha) * student_loss
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}"
                )

        logger.info("Knowledge distillation completed")
        return student_model

    def create_student_model(
        self, teacher_model: nn.Module, compression_ratio: float = 0.5
    ) -> nn.Module:
        """Create a smaller student model."""
        # Count parameters in teacher model
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        target_params = int(teacher_params * compression_ratio)

        # Create a smaller student model (simplified architecture)
        class StudentModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(StudentModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, output_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        # Determine student model size
        input_size = (
            teacher_model.fc1.in_features if hasattr(teacher_model, "fc1") else 784
        )
        output_size = (
            teacher_model.fc3.out_features if hasattr(teacher_model, "fc3") else 10
        )
        hidden_size = max(32, int(target_params / (input_size + output_size)))

        student_model = StudentModel(input_size, hidden_size, output_size)

        logger.info(
            f"Created student model with {sum(p.numel() for p in student_model.parameters())} parameters"
        )
        return student_model


class NeuralArchitectureSearch:
    """Neural Architecture Search for efficient models."""

    def __init__(self, config: CompressionConfig):
        self.config = config

    def search_efficient_architecture(
        self, input_shape: Tuple[int, ...], output_size: int, max_params: int = 100000
    ) -> nn.Module:
        """Search for efficient neural architecture."""
        logger.info("Starting neural architecture search")

        # Define search space
        hidden_sizes = [32, 64, 128, 256, 512]
        num_layers = [2, 3, 4, 5]
        activations = [nn.ReLU, nn.GELU, nn.Swish]

        best_model = None
        best_score = float("inf")

        for hidden_size in hidden_sizes:
            for num_layer in num_layers:
                for activation in activations:
                    # Create model
                    model = self._create_model(
                        input_shape, output_size, hidden_size, num_layer, activation
                    )

                    # Check parameter count
                    param_count = sum(p.numel() for p in model.parameters())
                    if param_count > max_params:
                        continue

                    # Evaluate model (simplified scoring)
                    score = self._evaluate_architecture(model)

                    if score < best_score:
                        best_score = score
                        best_model = model

        logger.info(
            f"Found best architecture with {sum(p.numel() for p in best_model.parameters())} parameters"
        )
        return best_model

    def _create_model(
        self,
        input_shape: Tuple[int, ...],
        output_size: int,
        hidden_size: int,
        num_layers: int,
        activation,
    ) -> nn.Module:
        """Create a model with specified architecture."""
        input_size = np.prod(input_shape)

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(hidden_size, output_size))

        return nn.Sequential(*layers)

    def _evaluate_architecture(self, model: nn.Module) -> float:
        """Evaluate architecture efficiency (simplified)."""
        # Simple scoring based on parameter count and model depth
        param_count = sum(p.numel() for p in model.parameters())
        depth = len(list(model.modules())) - 1  # Subtract 1 for the main module

        # Lower is better
        score = param_count * 0.001 + depth * 0.1
        return score


class ModelCompressor:
    """Main model compression orchestrator."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.quantizer = ModelQuantizer(config)
        self.pruner = ModelPruner(config)
        self.distillation = KnowledgeDistillation(config)
        self.nas = NeuralArchitectureSearch(config)

    def compress_model(
        self, model: nn.Module, calibration_data: torch.Tensor = None, train_loader=None
    ) -> Dict[str, Any]:
        """Apply comprehensive model compression."""
        logger.info("Starting model compression")

        original_size = sum(p.numel() for p in model.parameters())
        compressed_model = model
        compression_steps = []

        # Step 1: Pruning
        if self.config.enable_pruning:
            logger.info("Applying pruning")
            compressed_model = self.pruner.magnitude_pruning(compressed_model)
            pruned_size = sum(p.numel() for p in compressed_model.parameters())
            compression_steps.append(
                {
                    "step": "pruning",
                    "size_before": original_size,
                    "size_after": pruned_size,
                    "compression_ratio": pruned_size / original_size,
                }
            )

        # Step 2: Quantization
        if self.config.enable_quantization and calibration_data is not None:
            logger.info("Applying quantization")
            quantized_model = self.quantizer.static_quantization(
                compressed_model, calibration_data
            )
            quantized_size = sum(p.numel() for p in quantized_model.parameters())
            compression_steps.append(
                {
                    "step": "quantization",
                    "size_before": (
                        pruned_size if self.config.enable_pruning else original_size
                    ),
                    "size_after": quantized_size,
                    "compression_ratio": quantized_size / original_size,
                }
            )
            compressed_model = quantized_model

        # Step 3: Knowledge Distillation (if training data available)
        if self.config.enable_distillation and train_loader is not None:
            logger.info("Applying knowledge distillation")
            student_model = self.distillation.create_student_model(compressed_model)
            distilled_model = self.distillation.distill_knowledge(
                compressed_model, student_model, train_loader
            )
            distilled_size = sum(p.numel() for p in distilled_model.parameters())
            compression_steps.append(
                {
                    "step": "distillation",
                    "size_before": sum(
                        p.numel() for p in compressed_model.parameters()
                    ),
                    "size_after": distilled_size,
                    "compression_ratio": distilled_size / original_size,
                }
            )
            compressed_model = distilled_model

        final_size = sum(p.numel() for p in compressed_model.parameters())
        overall_compression = final_size / original_size

        compression_report = {
            "original_size": original_size,
            "final_size": final_size,
            "overall_compression_ratio": overall_compression,
            "compression_steps": compression_steps,
            "config": {
                "target_compression_ratio": self.config.target_compression_ratio,
                "quantization_bits": self.config.quantization_bits,
                "pruning_ratio": self.config.pruning_ratio,
            },
        }

        logger.info(
            f"Model compression completed. Final compression ratio: {overall_compression:.3f}"
        )
        return compressed_model, compression_report

    def benchmark_compressed_model(
        self, original_model: nn.Module, compressed_model: nn.Module, test_data
    ) -> Dict[str, Any]:
        """Benchmark compressed model against original."""
        logger.info("Benchmarking compressed model")

        # Size comparison
        original_size = sum(p.numel() for p in original_model.parameters())
        compressed_size = sum(p.numel() for p in compressed_model.parameters())

        # Performance comparison
        original_model.eval()
        compressed_model.eval()

        original_accuracy = self._evaluate_model(original_model, test_data)
        compressed_accuracy = self._evaluate_model(compressed_model, test_data)

        benchmark_results = {
            "size_comparison": {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compressed_size / original_size,
                "size_reduction": (original_size - compressed_size) / original_size,
            },
            "performance_comparison": {
                "original_accuracy": original_accuracy,
                "compressed_accuracy": compressed_accuracy,
                "accuracy_drop": original_accuracy - compressed_accuracy,
            },
            "efficiency_metrics": {
                "compression_efficiency": (original_size - compressed_size)
                / original_size,
                "accuracy_retention": (
                    compressed_accuracy / original_accuracy
                    if original_accuracy > 0
                    else 0
                ),
            },
        }

        return benchmark_results

    def _evaluate_model(self, model: nn.Module, test_data) -> float:
        """Evaluate model accuracy."""
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_data:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total if total > 0 else 0.0
