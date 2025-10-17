"""
Kafka-based Real-time Streaming Processor

Implements real-time data processing using Apache Kafka for handling
live data streams and real-time ML inference.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("Kafka libraries not installed. Install with: pip install kafka-python")
    KafkaProducer = None
    KafkaConsumer = None

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for Kafka streaming."""

    bootstrap_servers: List[str] = None
    group_id: str = "easylife-ai"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str = "PLAIN"
    sasl_username: str = None
    sasl_password: str = None


class KafkaStreamProcessor:
    """Kafka-based streaming processor for real-time ML inference."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.producer = None
        self.consumer = None
        self.processors = {}
        self.running = False

    def _create_producer(self) -> KafkaProducer:
        """Create Kafka producer."""
        if not KafkaProducer:
            raise ImportError("Kafka libraries not installed")

        producer_config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            "security_protocol": self.config.security_protocol,
        }

        if self.config.sasl_username and self.config.sasl_password:
            producer_config.update(
                {
                    "sasl_mechanism": self.config.sasl_mechanism,
                    "sasl_plain_username": self.config.sasl_username,
                    "sasl_plain_password": self.config.sasl_password,
                }
            )

        return KafkaProducer(**producer_config)

    def _create_consumer(self, topics: List[str]) -> KafkaConsumer:
        """Create Kafka consumer."""
        if not KafkaConsumer:
            raise ImportError("Kafka libraries not installed")

        consumer_config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "group_id": self.config.group_id,
            "auto_offset_reset": self.config.auto_offset_reset,
            "enable_auto_commit": self.config.enable_auto_commit,
            "max_poll_records": self.config.max_poll_records,
            "session_timeout_ms": self.config.session_timeout_ms,
            "heartbeat_interval_ms": self.config.heartbeat_interval_ms,
            "value_deserializer": lambda m: json.loads(m.decode("utf-8")),
            "key_deserializer": lambda m: m.decode("utf-8") if m else None,
            "security_protocol": self.config.security_protocol,
        }

        if self.config.sasl_username and self.config.sasl_password:
            consumer_config.update(
                {
                    "sasl_mechanism": self.config.sasl_mechanism,
                    "sasl_plain_username": self.config.sasl_username,
                    "sasl_plain_password": self.config.sasl_password,
                }
            )

        return KafkaConsumer(*topics, **consumer_config)

    def register_processor(self, topic: str, processor_func: Callable[[Dict], Dict]):
        """Register a processor function for a specific topic."""
        self.processors[topic] = processor_func
        logger.info(f"Registered processor for topic: {topic}")

    async def start_processing(self, topics: List[str]):
        """Start processing messages from Kafka topics."""
        if not self.processors:
            raise ValueError("No processors registered")

        self.consumer = self._create_consumer(topics)
        self.running = True

        logger.info(f"Starting Kafka processing for topics: {topics}")

        try:
            while self.running:
                message_batch = self.consumer.poll(timeout_ms=1000)

                for topic_partition, messages in message_batch.items():
                    topic = topic_partition.topic

                    if topic in self.processors:
                        processor = self.processors[topic]

                        for message in messages:
                            try:
                                # Process message
                                result = await self._process_message(processor, message)

                                # Send result to output topic if configured
                                if result:
                                    await self._send_result(topic, result, message.key)

                            except Exception as e:
                                logger.error(
                                    f"Error processing message from {topic}: {e}"
                                )
                                await self._handle_error(topic, message, e)

        except Exception as e:
            logger.error(f"Error in Kafka processing: {e}")
        finally:
            await self.stop_processing()

    async def _process_message(self, processor: Callable, message) -> Optional[Dict]:
        """Process a single message."""
        try:
            # Extract message data
            message_data = {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "timestamp": message.timestamp,
                "key": message.key,
                "value": message.value,
                "headers": dict(message.headers) if message.headers else {},
            }

            # Process with registered function
            result = processor(message_data)

            return result

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None

    async def _send_result(self, topic: str, result: Dict, key: Optional[str] = None):
        """Send processing result to output topic."""
        if not self.producer:
            self.producer = self._create_producer()

        output_topic = f"{topic}_processed"

        try:
            self.producer.send(output_topic, value=result, key=key)
            self.producer.flush()
            logger.debug(f"Sent result to {output_topic}")
        except KafkaError as e:
            logger.error(f"Error sending result to {output_topic}: {e}")

    async def _handle_error(self, topic: str, message, error: Exception):
        """Handle processing errors."""
        error_topic = f"{topic}_errors"

        error_data = {
            "original_topic": topic,
            "error_message": str(error),
            "error_timestamp": datetime.utcnow().isoformat(),
            "original_message": {
                "key": message.key,
                "value": message.value,
                "offset": message.offset,
                "partition": message.partition,
            },
        }

        if not self.producer:
            self.producer = self._create_producer()

        try:
            self.producer.send(error_topic, value=error_data)
            self.producer.flush()
            logger.info(f"Sent error to {error_topic}")
        except KafkaError as e:
            logger.error(f"Error sending error message: {e}")

    async def stop_processing(self):
        """Stop Kafka processing."""
        self.running = False

        if self.consumer:
            self.consumer.close()

        if self.producer:
            self.producer.close()

        logger.info("Kafka processing stopped")


class RealTimeMLProcessor:
    """Real-time ML processor for streaming data."""

    def __init__(self, kafka_processor: KafkaStreamProcessor):
        self.kafka_processor = kafka_processor
        self.models = {}
        self.preprocessors = {}

    def register_model(
        self, model_name: str, model, preprocessor: Optional[Callable] = None
    ):
        """Register a model for real-time inference."""
        self.models[model_name] = model
        if preprocessor:
            self.preprocessors[model_name] = preprocessor

        logger.info(f"Registered model: {model_name}")

    def create_nlp_processor(self, topic: str = "nlp_input"):
        """Create NLP sentiment analysis processor."""

        def nlp_processor(message_data: Dict) -> Dict:
            try:
                text = message_data["value"].get("text", "")

                # Simulate NLP processing (replace with actual model)
                sentiment = "positive" if len(text) > 10 else "negative"
                confidence = 0.85

                result = {
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "input_text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "model_version": "1.0.0",
                }

                return result

            except Exception as e:
                logger.error(f"NLP processing error: {e}")
                return None

        self.kafka_processor.register_processor(topic, nlp_processor)
        logger.info(f"Created NLP processor for topic: {topic}")

    def create_cv_processor(self, topic: str = "cv_input"):
        """Create CV image analysis processor."""

        def cv_processor(message_data: Dict) -> Dict:
            try:
                # image_data = message_data["value"].get("image", "")  # Not used in simulation

                # Simulate CV processing (replace with actual model)
                blur_score = 0.3
                is_blurry = blur_score > 0.5

                result = {
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "blur_score": blur_score,
                    "is_blurry": is_blurry,
                    "confidence": 0.92,
                    "model_version": "1.0.0",
                }

                return result

            except Exception as e:
                logger.error(f"CV processing error: {e}")
                return None

        self.kafka_processor.register_processor(topic, cv_processor)
        logger.info(f"Created CV processor for topic: {topic}")

    def create_ts_processor(self, topic: str = "ts_input"):
        """Create time series forecasting processor."""

        def ts_processor(message_data: Dict) -> Dict:
            try:
                # time_series = message_data["value"].get("data", [])  # Not used in simulation
                horizon = message_data["value"].get("horizon", 7)

                # Simulate time series forecasting (replace with actual model)
                forecast = [100 + i * 2 for i in range(horizon)]

                result = {
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "forecast": forecast,
                    "horizon": horizon,
                    "confidence": 0.88,
                    "model_version": "1.0.0",
                }

                return result

            except Exception as e:
                logger.error(f"Time series processing error: {e}")
                return None

        self.kafka_processor.register_processor(topic, ts_processor)
        logger.info(f"Created time series processor for topic: {topic}")


class StreamingManager:
    """Manages real-time streaming operations."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.kafka_processor = KafkaStreamProcessor(config)
        self.ml_processor = RealTimeMLProcessor(self.kafka_processor)

    async def start_streaming(self, topics: List[str]):
        """Start real-time streaming processing."""
        logger.info(f"Starting streaming processing for topics: {topics}")

        # Create processors for each topic
        for topic in topics:
            if topic == "nlp_input":
                self.ml_processor.create_nlp_processor(topic)
            elif topic == "cv_input":
                self.ml_processor.create_cv_processor(topic)
            elif topic == "ts_input":
                self.ml_processor.create_ts_processor(topic)

        # Start processing
        await self.kafka_processor.start_processing(topics)

    async def stop_streaming(self):
        """Stop streaming processing."""
        await self.kafka_processor.stop_processing()
        logger.info("Streaming processing stopped")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "registered_processors": len(self.kafka_processor.processors),
            "registered_models": len(self.ml_processor.models),
            "is_running": self.kafka_processor.running,
            "config": {
                "bootstrap_servers": self.config.bootstrap_servers,
                "group_id": self.config.group_id,
                "max_poll_records": self.config.max_poll_records,
            },
        }
