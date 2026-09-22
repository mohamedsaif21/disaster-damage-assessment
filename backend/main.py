from fastapi import FastAPI


app = FastAPI(
    title="AI Disaster Damage Assessment API",
    description="Backend API for AI-powered disaster damage assessment.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "AI Disaster Damage Assessment API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }