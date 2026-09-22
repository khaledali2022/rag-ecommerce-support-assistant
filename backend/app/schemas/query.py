from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural-language question.")


class SourceChunk(BaseModel):
    document: str
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: list[SourceChunk] = []


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    collection_count: int
