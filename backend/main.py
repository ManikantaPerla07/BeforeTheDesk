import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import (
    ALLOWED_ORIGINS,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    SPACY_MODEL_PRIMARY,
    SPACY_MODEL_SECONDARY,
    SENTENCE_TRANSFORMER_MODEL,
)
from backend.api.routes import router

logger = logging.getLogger("ats_resume_scorer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown.
    Heavy AI models are NOT loaded here.
    They will be loaded only when the first resume is analyzed.
    """
    logger.info("Starting ATS Resume Analyzer API...")

    # Lazy-loaded models
    app.state.nlp = None
    app.state.embedder = None

    logger.info("Lazy loading enabled. Models will load on first request.")

    yield

    logger.info("Shutting down ATS Resume Analyzer API...")


def load_models(app: FastAPI):
    """
    Load NLP models only once.
    Safe to call on every request.
    """

    # Load spaCy only once
    if app.state.nlp is None:
        logger.info(f"Loading spaCy model: {SPACY_MODEL_PRIMARY}")

        import spacy

        try:
            app.state.nlp = spacy.load(SPACY_MODEL_PRIMARY)
            logger.info(f"Loaded {SPACY_MODEL_PRIMARY}")
        except OSError:
            logger.warning(
                f"{SPACY_MODEL_PRIMARY} not found. Falling back to {SPACY_MODEL_SECONDARY}"
            )
            app.state.nlp = spacy.load(SPACY_MODEL_SECONDARY)
            logger.info(f"Loaded {SPACY_MODEL_SECONDARY}")

    # Load SentenceTransformer only once
    if app.state.embedder is None:
        logger.info(f"Loading SentenceTransformer: {SENTENCE_TRANSFORMER_MODEL}")

        from sentence_transformers import SentenceTransformer

        app.state.embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL
        )

        logger.info(f"Loaded {SENTENCE_TRANSFORMER_MODEL}")

    return app.state.nlp, app.state.embedder


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "ATS Resume Analyzer API",
        "version": "2.0.0",
        "endpoints": {
            "POST   /api/v1/analyze-resume": "Analyze a resume",
            "GET    /api/v1/history": "Get user history",
            "DELETE /api/v1/history/:id": "Delete a history entry",
            "GET    /api/v1/health": "Health check",
            "POST   /api/v1/generate-pdf": "Generate PDF report from data",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )