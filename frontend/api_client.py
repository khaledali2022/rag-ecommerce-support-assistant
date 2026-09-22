"""
Thin wrapper around the FastAPI backend. The backend URL is always read
from an environment variable -- never hard-coded.
"""
import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 30


class ApiError(Exception):
    pass


def check_health() -> dict:
    response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def ask_question(question: str) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Could not reach the backend at {API_BASE_URL}. Is it running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The backend took too long to respond. Please try again.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = response.text
        raise ApiError(f"Backend returned an error: {detail}") from exc
