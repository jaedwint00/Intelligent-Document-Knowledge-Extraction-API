"""
Intelligent Document & Knowledge Extraction API
Main FastAPI application entry point
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import document, nlp, search
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.nlp_service import NLPService
from app.services.vector_service import VectorService


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Intelligent Document & Knowledge Extraction API")

    # Initialize services
    nlp_service = NLPService()
    vector_service = VectorService()

    # Store services in app state
    fastapi_app.state.nlp_service = nlp_service
    fastapi_app.state.vector_service = vector_service

    logger.info("Services initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    # Setup logging first
    setup_logging()

    fastapi_app = FastAPI(
        title="Intelligent Document & Knowledge Extraction API",
        description=(
            "Multi-language AI service for extracting insights, summaries, "
            "and structured knowledge from unstructured documents"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    fastapi_app.include_router(
        document.router, prefix="/api/v1/documents", tags=["documents"]
    )
    fastapi_app.include_router(nlp.router, prefix="/api/v1/nlp", tags=["nlp"])
    fastapi_app.include_router(search.router, prefix="/api/v1/search", tags=["search"])

    @fastapi_app.get("/")
    async def root():
        return {
            "message": "Intelligent Document & Knowledge Extraction API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    @fastapi_app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": settings.get_current_timestamp()}

    return fastapi_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
