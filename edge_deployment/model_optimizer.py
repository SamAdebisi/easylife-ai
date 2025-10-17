"""
Edge Model Optimizer

Optimizes models for edge deployment with quantization, pruning, and
mobile-specific optimizations for IoT and mobile devices.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EdgeOptimizationConfig:
    """Configuration for edge model optimization."""

    target_device: str = "mobile"  # mobile, iot, embedded
    quantization_bits: int = 8
    pruning_ratio: float = 0.3
    optimize_for_inference: bool = True
    enable_tensorrt: bool = False
    memory_budget_mb: int = 100
    latency_target_ms: float = 50.0


class ModelQuantizer:
    """Quantizes models for edge deployment."""

    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config

    def quantize_model(
        self, model: nn.Module, calibration_data: torch.Tensor
    ) -> nn.Module:
        """Quantize model for edge deployment."""
        logger.info(f"Quantizing model to {self.config.quantization_bits} bits")

        # Set model to evaluation mode
        model.eval()

        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # Calibrate with sample data
        with torch.no_grad():
            for i in range(min(100, len(calibration_data))):
                sample = calibration_data[i : i + 1]
                _ = model_prepared(sample)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)

        logger.info("Model quantization completed")
        return quantized_model

    def dynamic_quantization(
        self, model: nn.Module, modules_to_quantize: List[nn.Module] = None
    ) -> nn.Module:
        """Apply dynamic quantization."""
        logger.info("Applying dynamic quantization")
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )

    def static_quantization(
        self, model: nn.Module, calibration_data: torch.Tensor
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        logger.info("Applying static quantization")
        return self.quantize_model(model, calibration_data)


class ModelPruner:
    """Prunes models to reduce size and improve inference speed."""

    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config

    def magnitude_pruning(self, model: nn.Module) -> nn.Module:
        """Apply magnitude-based pruning."""
        logger.info(
            f"Applying magnitude pruning with ratio {self.config.pruning_ratio}"
        )

        # Get all parameters
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


class MobileOptimizer:
    """Mobile-specific optimizations."""

    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config

    def optimize_for_mobile(self, model: nn.Module) -> nn.Module:
        """Optimize model for mobile deployment."""
        logger.info("Optimizing model for mobile deployment")

        # Convert to TorchScript for mobile
        model.eval()
        scripted_model = torch.jit.script(model)

        # Optimize for mobile
        optimized_model = torch.jit.optimize_for_inference(scripted_model)

        logger.info("Mobile optimization completed")
        return optimized_model

    def optimize_for_iot(self, model: nn.Module) -> nn.Module:
        """Optimize model for IoT devices."""
        logger.info("Optimizing model for IoT deployment")

        # Apply aggressive optimizations for IoT
        model.eval()

        # Use half precision if supported
        if self.config.target_device == "iot":
            model = model.half()

        # Convert to TorchScript
        scripted_model = torch.jit.script(model)

        logger.info("IoT optimization completed")
        return scripted_model


class EdgeModelOptimizer:
    """Main edge model optimizer."""

    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config
        self.quantizer = ModelQuantizer(config)
        self.pruner = ModelPruner(config)
        self.mobile_optimizer = MobileOptimizer(config)

    def optimize_model(
        self, model: nn.Module, calibration_data: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Optimize model for edge deployment."""
        logger.info("Starting edge model optimization")

        # Step 1: Pruning
        if self.config.pruning_ratio > 0:
            model = self.pruner.magnitude_pruning(model)

        # Step 2: Quantization
        if self.config.quantization_bits < 32:
            if calibration_data is not None:
                model = self.quantizer.static_quantization(model, calibration_data)
            else:
                model = self.quantizer.dynamic_quantization(model)

        # Step 3: Device-specific optimization
        if self.config.target_device == "mobile":
            model = self.mobile_optimizer.optimize_for_mobile(model)
        elif self.config.target_device == "iot":
            model = self.mobile_optimizer.optimize_for_iot(model)

        logger.info("Edge model optimization completed")
        return model

    def get_model_stats(self, model: nn.Module) -> Dict[str, Union[int, float]]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "compression_ratio": 1.0,  # Would be calculated based on original model
        }

    def benchmark_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        import time

        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = np.mean(times)
        std_latency = np.std(times)
        p95_latency = np.percentile(times, 95)

        return {
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "p95_latency_ms": p95_latency,
            "throughput_fps": 1000 / avg_latency,
        }


class EdgeDeploymentManager:
    """Manages edge deployment of optimized models."""

    def __init__(self, config: EdgeOptimizationConfig):
        self.config = config
        self.optimizer = EdgeModelOptimizer(config)
        self.deployed_models = {}

    def deploy_model(
        self,
        model: nn.Module,
        model_name: str,
        calibration_data: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> Dict[str, Union[str, Dict]]:
        """Deploy optimized model to edge device."""
        logger.info(f"Deploying model {model_name} to edge device")

        # Optimize model
        optimized_model = self.optimizer.optimize_model(model, calibration_data)

        # Get model statistics
        stats = self.optimizer.get_model_stats(optimized_model)

        # Benchmark model
        input_shape = (3, 224, 224)  # Default for image models
        benchmark_results = self.optimizer.benchmark_model(optimized_model, input_shape)

        # Store deployment info
        deployment_info = {
            "model_name": model_name,
            "optimized_model": optimized_model,
            "stats": stats,
            "benchmark": benchmark_results,
            "config": self.config,
            "deployment_time": torch.tensor(
                [
                    (
                        torch.cuda.Event(enable_timing=True).elapsed_time()
                        if torch.cuda.is_available()
                        else 0
                    )
                ]
            ),
        }

        self.deployed_models[model_name] = deployment_info

        logger.info(f"Model {model_name} deployed successfully")
        return deployment_info

    def get_deployment_status(self) -> Dict[str, Dict]:
        """Get status of all deployed models."""
        status = {}
        for model_name, info in self.deployed_models.items():
            status[model_name] = {
                "model_size_mb": info["stats"]["model_size_mb"],
                "avg_latency_ms": info["benchmark"]["avg_latency_ms"],
                "throughput_fps": info["benchmark"]["throughput_fps"],
                "target_device": self.config.target_device,
            }
        return status
