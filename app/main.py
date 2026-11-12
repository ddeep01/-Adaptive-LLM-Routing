"""
Adaptive LLM Routing - FastAPI Application Entrypoint
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.api.routes import router
from app.core.logger import logger

app = FastAPI(
    title="Adaptive LLM Routing API",
    description="Intelligent query complexity estimation and cost/latency optimizing LLM routing framework.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response

# Include API routes
app.include_router(router)

@app.get("/", tags=["Root"])
def root():
    return {
        "title": "Adaptive LLM Routing System",
        "documentation": "/docs",
        "health": "/health",
        "models": "/models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
