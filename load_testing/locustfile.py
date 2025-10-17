"""
Load testing for EasyLife AI services using Locust.
"""

import random

from locust import HttpUser, between, task


class NLPServiceUser(HttpUser):
    """Load test for NLP sentiment analysis service."""

    wait_time = between(1, 3)
    host = "http://localhost:8001"

    def on_start(self):
        """Setup test data."""
        self.test_texts = [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Terrible experience, would not recommend to anyone.",
            "The product is okay, nothing special but gets the job done.",
            "Amazing quality and fast delivery, will definitely buy again!",
            "Waste of money, poor quality and slow service.",
            "Great value for money, exceeded my expectations.",
            "Average product, nothing to write home about.",
            "Outstanding customer service and excellent product quality!",
        ]

    @task(3)
    def predict_sentiment(self):
        """Test sentiment prediction endpoint."""
        text = random.choice(self.test_texts)
        payload = {"text": text}

        with self.client.post(
            "/predict", json=payload, catch_response=True, name="predict_sentiment"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "label" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="health_check")

    @task(1)
    def metrics_endpoint(self):
        """Test metrics endpoint."""
        self.client.get("/metrics", name="metrics")


class CVServiceUser(HttpUser):
    """Load test for Computer Vision blur detection service."""

    wait_time = between(2, 5)
    host = "http://localhost:8002"

    @task(2)
    def predict_image(self):
        """Test image prediction endpoint."""
        # Create a simple test image (1x1 pixel)
        import io

        from PIL import Image

        img = Image.new("RGB", (1, 1), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        with self.client.post(
            "/predict", files=files, catch_response=True, name="predict_image"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "blur_score" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="health_check")


class TSForecastingUser(HttpUser):
    """Load test for Time Series Forecasting service."""

    wait_time = between(3, 8)
    host = "http://localhost:8003"

    @task(3)
    def forecast(self):
        """Test forecasting endpoint."""
        horizon = random.randint(7, 30)

        with self.client.get(
            f"/forecast?horizon={horizon}", catch_response=True, name="forecast"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "forecast" in data and "confidence_interval" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="health_check")


class RecsysServiceUser(HttpUser):
    """Load test for Recommendation System service."""

    wait_time = between(1, 4)
    host = "http://localhost:8004"

    def on_start(self):
        """Setup test data."""
        self.user_ids = [f"user_{i}" for i in range(1, 101)]
        self.item_ids = [f"item_{i}" for i in range(1, 201)]

    @task(3)
    def get_recommendations(self):
        """Test recommendation endpoint."""
        user_id = random.choice(self.user_ids)
        top_k = random.choice([5, 10, 20])

        with self.client.get(
            f"/recommend?user_id={user_id}&top_k={top_k}",
            catch_response=True,
            name="get_recommendations",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "recommendations" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def get_similar_items(self):
        """Test similar items endpoint."""
        item_id = random.choice(self.item_ids)
        top_k = random.choice([5, 10])

        with self.client.get(
            f"/similar?item_id={item_id}&top_k={top_k}",
            catch_response=True,
            name="get_similar_items",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "similar_items" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="health_check")


class EasyLifeAIScenario(HttpUser):
    """End-to-end scenario testing."""

    wait_time = between(5, 15)

    def on_start(self):
        """Setup for end-to-end testing."""
        self.nlp_host = "http://localhost:8001"
        self.cv_host = "http://localhost:8002"
        self.ts_host = "http://localhost:8003"
        self.recsys_host = "http://localhost:8004"

    @task(1)
    def end_to_end_workflow(self):
        """Simulate a complete user workflow."""
        # 1. Analyze sentiment of a review
        sentiment_payload = {"text": "Great product, highly recommended!"}
        self.client.post(
            f"{self.nlp_host}/predict", json=sentiment_payload, name="e2e_sentiment"
        )

        # 2. Get recommendations
        self.client.get(
            f"{self.recsys_host}/recommend?user_id=user_1&top_k=5",
            name="e2e_recommendations",
        )

        # 3. Get forecast
        self.client.get(f"{self.ts_host}/forecast?horizon=7", name="e2e_forecast")

        # 4. Check system health
        self.client.get(f"{self.nlp_host}/health", name="e2e_health")
        self.client.get(f"{self.cv_host}/health", name="e2e_health")
        self.client.get(f"{self.ts_host}/health", name="e2e_health")
        self.client.get(f"{self.recsys_host}/health", name="e2e_health")
