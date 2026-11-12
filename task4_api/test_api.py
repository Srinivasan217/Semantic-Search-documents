
from fastapi.testclient import TestClient

from task4_api.main import app, API_KEY


client_headers = {"x-api-key": API_KEY}


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200


def test_search_without_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/search",
            json={"query": "password reset", "top_k": 5},
        )

        assert response.status_code == 401


def test_search_with_invalid_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/search",
            headers={"x-api-key": "wrong-key"},
            json={"query": "password reset", "top_k": 5},
        )

        assert response.status_code == 401


def test_search_with_valid_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/search",
            headers=client_headers,
            json={"query": "password reset policy", "top_k": 5},
        )

        assert response.status_code == 200
        assert "results" in response.json()


def test_search_with_document_type_filter():
    with TestClient(app) as client:
        response = client.post(
            "/search",
            headers=client_headers,
            json={
                "query": "document upload formats",
                "top_k": 5,
                "doc_type": "txt",
                "min_score": 0.0,
            },
        )

        assert response.status_code == 200
        assert response.json()["filters"]["doc_type"] == "txt"


def test_upload_rejects_unsupported_file():
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            headers=client_headers,
            files={
                "file": (
                    "sample.exe",
                    b"test content",
                    "application/octet-stream",
                )
            },
        )

        assert response.status_code == 400
