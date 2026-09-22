import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse, SourceChunk
from app.services import generation
from app.services.retrieval import retrieval_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if retrieval_service.is_loaded else "vector_store_not_loaded",
        vector_store_loaded=retrieval_service.is_loaded,
        collection_count=retrieval_service.count,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not retrieval_service.is_loaded:
        raise HTTPException(status_code=503, detail="Vector store is not loaded yet.")

    try:
        chunks = retrieval_service.retrieve(request.question)
        answer = generation.generate_answer(request.question, chunks)
    except RuntimeError as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = sorted({c["document"] for c in chunks})

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=[
            SourceChunk(
                document=c["document"],
                chunk_id=c["chunk_id"],
                text=c["text"],
                score=c["score"],
            )
            for c in chunks
        ],
    )
