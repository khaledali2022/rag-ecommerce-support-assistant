"""
Tests for the /query and /health endpoints.

Run with:  pytest -v

These tests use FastAPI's TestClient, which triggers the app's lifespan
(loading the persisted vector store) on startup, so make sure the vector
store has already been built by the notebook before running them.
"""
from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "vector_store_loaded" in body


def test_query_happy_path():
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is your return policy?"})
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0
        assert "sources" in body
        assert isinstance(body["sources"], list)


def test_query_invalid_input_returns_422():
    with TestClient(app) as client:
        # Missing the required "question" field entirely
        response = client.post("/query", json={})
        assert response.status_code == 422


def test_query_empty_question_returns_422():
    with TestClient(app) as client:
        # Empty string violates min_length=1
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422
