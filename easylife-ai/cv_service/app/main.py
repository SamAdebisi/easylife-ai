from fastapi import FastAPI

app = FastAPI(title="EasyLife AI Service", version="0.1.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
