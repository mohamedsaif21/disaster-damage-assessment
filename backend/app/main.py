from fastapi import FastAPI

app = FastAPI(
    title="AI Disaster Damage Assessment API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "disaster-damage-assessment-api",
        "version": "0.1.0",
    }