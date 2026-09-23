from fastapi import FastAPI

from backend.api.assessment import router as assessment_router


app = FastAPI(
    title="AI Disaster Damage Assessment API",
    description="Backend API for AI-powered disaster damage assessment.",
    version="1.0.0"
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    assessment_router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Disaster Damage Assessment API",
        "status": "running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }