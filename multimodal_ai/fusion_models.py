"""
Multi-modal Fusion Models

Implements multi-modal AI models that can process and fuse text, image,
and time series data for comprehensive analysis and prediction.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal models."""

    text_embedding_dim: int = 768
    image_embedding_dim: int = 2048
    time_series_embedding_dim: int = 128
    fusion_dim: int = 512
    num_classes: int = 10
    dropout_rate: float = 0.3
    attention_heads: int = 8


class TextEncoder(nn.Module):
    """Text encoder for multi-modal models."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 2
    ):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, text_input):
        # text_input: (batch_size, seq_length)
        embedded = self.embedding(text_input)  # (batch_size, seq_length, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state
        text_features = hidden[-1]  # (batch_size, hidden_dim)
        text_features = self.dropout(text_features)

        return text_features


class ImageEncoder(nn.Module):
    """Image encoder for multi-modal models."""

    def __init__(self, input_channels: int = 3, embedding_dim: int = 2048):
        super(ImageEncoder, self).__init__()

        # CNN backbone
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-like blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, image_input):
        # image_input: (batch_size, channels, height, width)
        x = self.conv1(image_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)

        return x


class TimeSeriesEncoder(nn.Module):
    """Time series encoder for multi-modal models."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, time_series_input):
        # time_series_input: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(time_series_input)

        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        time_series_features = torch.mean(attended_out, dim=1)
        time_series_features = self.dropout(time_series_features)

        return time_series_features


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""

    def __init__(self, dim: int, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        # Cross-modal attention
        attended, _ = self.attention(query, key, value)
        output = self.norm(attended + query)
        return output


class MultiModalFusion(nn.Module):
    """Multi-modal fusion network."""

    def __init__(self, config: MultiModalConfig):
        super(MultiModalFusion, self).__init__()
        self.config = config

        # Projection layers to common dimension
        self.text_projection = nn.Linear(config.text_embedding_dim, config.fusion_dim)
        self.image_projection = nn.Linear(config.image_embedding_dim, config.fusion_dim)
        self.time_series_projection = nn.Linear(
            config.time_series_embedding_dim, config.fusion_dim
        )

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            config.fusion_dim, config.attention_heads
        )

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(config.fusion_dim * 3, config.fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )

        # Classification head
        self.classifier = nn.Linear(config.fusion_dim, config.num_classes)

    def forward(self, text_features, image_features, time_series_features):
        # Project to common dimension
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        time_series_proj = self.time_series_projection(time_series_features)

        # Cross-modal attention
        # Text attends to image and time series
        text_attended = self.cross_attention(
            text_proj.unsqueeze(1),
            torch.stack([image_proj, time_series_proj], dim=1),
            torch.stack([image_proj, time_series_proj], dim=1),
        ).squeeze(1)

        # Image attends to text and time series
        image_attended = self.cross_attention(
            image_proj.unsqueeze(1),
            torch.stack([text_proj, time_series_proj], dim=1),
            torch.stack([text_proj, time_series_proj], dim=1),
        ).squeeze(1)

        # Time series attends to text and image
        time_series_attended = self.cross_attention(
            time_series_proj.unsqueeze(1),
            torch.stack([text_proj, image_proj], dim=1),
            torch.stack([text_proj, image_proj], dim=1),
        ).squeeze(1)

        # Concatenate attended features
        fused_features = torch.cat(
            [text_attended, image_attended, time_series_attended], dim=1
        )

        # Apply fusion layers
        fused_features = self.fusion_layers(fused_features)

        # Classification
        output = self.classifier(fused_features)

        return output


class MultiModalModel(nn.Module):
    """Complete multi-modal model."""

    def __init__(
        self, config: MultiModalConfig, vocab_size: int, time_series_input_dim: int
    ):
        super(MultiModalModel, self).__init__()
        self.config = config

        # Individual encoders
        self.text_encoder = TextEncoder(
            vocab_size, config.text_embedding_dim, config.text_embedding_dim
        )
        self.image_encoder = ImageEncoder(3, config.image_embedding_dim)
        self.time_series_encoder = TimeSeriesEncoder(
            time_series_input_dim, config.time_series_embedding_dim
        )

        # Fusion network
        self.fusion = MultiModalFusion(config)

    def forward(self, text_input, image_input, time_series_input):
        # Encode each modality
        text_features = self.text_encoder(text_input)
        image_features = self.image_encoder(image_input)
        time_series_features = self.time_series_encoder(time_series_input)

        # Fuse modalities
        output = self.fusion(text_features, image_features, time_series_features)

        return output


class MultiModalTrainer:
    """Trainer for multi-modal models."""

    def __init__(self, model: MultiModalModel, config: MultiModalConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (text_data, image_data, time_series_data, labels) in enumerate(
            train_loader
        ):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_data, image_data, time_series_data)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        return avg_loss, accuracy

    def evaluate(self, test_loader):
        """Evaluate model on test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for text_data, image_data, time_series_data, labels in test_loader:
                outputs = self.model(text_data, image_data, time_series_data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)

        return avg_loss, accuracy


class MultiModalInference:
    """Inference engine for multi-modal models."""

    def __init__(self, model: MultiModalModel):
        self.model = model
        self.model.eval()

    def predict(self, text_input, image_input, time_series_input):
        """Make prediction on multi-modal input."""
        with torch.no_grad():
            outputs = self.model(text_input, image_input, time_series_input)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        return predicted_class, probabilities

    def get_feature_importance(self, text_input, image_input, time_series_input):
        """Get feature importance for each modality."""
        # Get individual encoder outputs
        text_features = self.model.text_encoder(text_input)
        image_features = self.model.image_encoder(image_input)
        time_series_features = self.model.time_series_encoder(time_series_input)

        # Calculate feature importance (simplified)
        text_importance = torch.norm(text_features, dim=1)
        image_importance = torch.norm(image_features, dim=1)
        time_series_importance = torch.norm(time_series_features, dim=1)

        # Normalize
        total_importance = text_importance + image_importance + time_series_importance
        text_importance = text_importance / total_importance
        image_importance = image_importance / total_importance
        time_series_importance = time_series_importance / total_importance

        return {
            "text_importance": text_importance.item(),
            "image_importance": image_importance.item(),
            "time_series_importance": time_series_importance.item(),
        }


class MultiModalManager:
    """Manages multi-modal AI operations."""

    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.inference_engine = None

    def create_model(
        self, vocab_size: int, time_series_input_dim: int
    ) -> MultiModalModel:
        """Create multi-modal model."""
        self.model = MultiModalModel(self.config, vocab_size, time_series_input_dim)
        self.trainer = MultiModalTrainer(self.model, self.config)
        self.inference_engine = MultiModalInference(self.model)

        logger.info("Multi-modal model created successfully")
        return self.model

    def train_model(self, train_loader, test_loader, epochs: int = 10):
        """Train multi-modal model."""
        logger.info(f"Starting training for {epochs} epochs")

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.trainer.train_epoch(train_loader)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Evaluate
            test_loss, test_acc = self.trainer.evaluate(test_loader)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            logger.info(
                f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
            )

        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
        }

    def predict(self, text_input, image_input, time_series_input):
        """Make prediction using multi-modal model."""
        if self.inference_engine is None:
            raise ValueError("Model not created. Call create_model first.")

        return self.inference_engine.predict(text_input, image_input, time_series_input)

    def get_modality_importance(self, text_input, image_input, time_series_input):
        """Get importance of each modality."""
        if self.inference_engine is None:
            raise ValueError("Model not created. Call create_model first.")

        return self.inference_engine.get_feature_importance(
            text_input, image_input, time_series_input
        )

    def save_model(self, filepath: str):
        """Save multi-modal model."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save(
            {"model_state_dict": self.model.state_dict(), "config": self.config},
            filepath,
        )

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load multi-modal model."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.inference_engine = MultiModalInference(self.model)

        logger.info(f"Model loaded from {filepath}")
