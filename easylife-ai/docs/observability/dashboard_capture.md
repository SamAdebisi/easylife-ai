# Dashboard Capture Playbook

Follow this checklist after you bring the monitoring stack online (`make up`) so the documentation can showcase live telemetry.

1. **Start Infra**
   - `make up` to launch MinIO, MLflow, Prometheus, and Grafana.
   - Run the NLP pipeline: `dvc repro nlp-preprocess nlp-train`.
   - Run the CV pipeline: `dvc repro cv-ingest cv-preprocess cv-train cv-train-cnn`.
   - Start the FastAPI services locally:
     - NLP: `./nlp_service/run_dev.sh` or `uvicorn nlp_service.app.main:app --reload`.
     - CV: `./cv_service/run_dev.sh` or `uvicorn cv_service.app.main:app --reload --port 8002`.

2. **Generate Activity**
   - NLP: send at least 10–20 `POST /predict` requests covering positive and negative sentences. Example:
     ```bash
     curl -s http://localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"text": "I absolutely love the new update"}'
     ```
   - Watch the terminal to confirm predictions and confidence scores.
   - CV: upload a mix of sharp/blurred images via multipart requests.
     ```bash
     curl -s -X POST http://localhost:8002/predict_image \
       -F "file=@img/sample_sharp.png"
     ```
   - Confirm responses and check `/metrics` for `cv_predictions_total` increments (and verify `sharp_probability` entries if the CNN variant is active).

3. **Capture MLflow Screens**
   - Open `http://localhost:5000`.
   - Screenshot **Experiments → nlp_sentiment** and **nlp_service_inference** to highlight parameters, metrics, and confidence trends.
   - Capture **Experiments → cv_blur_detection** (threshold) and **cv_blur_detection_cnn** (CNN) to highlight accuracy trends, plus **cv_service_inference** to visualise `blur_score` or `sharp_probability` telemetry.

4. **Capture Grafana Panels**
   - Visit `http://localhost:3000` (credentials `admin` / `admin` by default).
   - Add Prometheus panels that chart `nlp_predictions_total` and `cv_predictions_total`/`sharp_probability` over time by label.
   - Export the panel as PNG (Panel menu → Share → Export) and save under `docs/assets/grafana-<date>.png`.

5. **Document**
   - Store exported images inside `docs/assets/`.
   - Reference them in upcoming documentation (e.g., `docs/model_cards/nlp.md`) using relative paths.
   - Update the changelog or README when new dashboards are captured.

Tips:
- Keep filenames timestamped (`mlflow-run-2024-10-05.png`) for easy diffing.
- Use the Grafana time picker to zoom into the prediction burst captured in step 2 for a cleaner screenshot.
