# Dashboard Capture Playbook

Follow this checklist after you bring the monitoring stack online (`make up`) so the documentation can showcase live telemetry.

1. **Start Infra**
   - `make up` to launch MinIO, MLflow, Prometheus, and Grafana.
   - Run the NLP pipeline: `dvc repro nlp-preprocess nlp-train`.
   - Start the FastAPI service locally: `./nlp_service/run_dev.sh` or `uvicorn nlp_service.app.main:app --reload`.

2. **Generate Activity**
   - Send at least 10–20 `POST /predict` requests covering positive and negative sentences. Example:
     ```bash
     curl -s http://localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"text": "I absolutely love the new update"}'
     ```
   - Watch the terminal to confirm predictions and confidence scores.

3. **Capture MLflow Screens**
   - Open `http://localhost:5000`.
   - Screenshot the experiment run in **Experiments → nlp_sentiment**, highlighting parameters and metrics.
   - If the inference telemetry run appears under **Experiments → nlp_service_inference**, capture the metrics chart showing `prediction_confidence`.

4. **Capture Grafana Panels**
   - Visit `http://localhost:3000` (credentials `admin` / `admin` by default).
   - Add a Prometheus panel that charts `nlp_predictions_total` by label.
   - Export the panel as PNG (Panel menu → Share → Export) and save under `docs/assets/grafana-<date>.png`.

5. **Document**
   - Store exported images inside `docs/assets/`.
   - Reference them in upcoming documentation (e.g., `docs/model_cards/nlp.md`) using relative paths.
   - Update the changelog or README when new dashboards are captured.

Tips:
- Keep filenames timestamped (`mlflow-run-2024-10-05.png`) for easy diffing.
- Use the Grafana time picker to zoom into the prediction burst captured in step 2 for a cleaner screenshot.
