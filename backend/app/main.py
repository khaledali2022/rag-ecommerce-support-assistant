import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import retrieval_service
from app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the vector store + embedding model ONCE, not per request.
    logger.info("Starting up %s ...", settings.APP_NAME)
    retrieval_service.load()
    yield
    # Shutdown (nothing to clean up for now)
    logger.info("Shutting down %s ...", settings.APP_NAME)


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, tags=["rag"])


@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running. See /docs for the API."}
