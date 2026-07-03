"""
tests/backend/test_health.py
─────────────────────────────
Integration tests for the /health endpoint.
"""

import pytest
from fastapi.testclient import TestClient

# Add the monorepo root to sys.path so imports work from /tests
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self) -> None:
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"

    def test_health_returns_service_name(self) -> None:
        response = client.get("/health")
        body = response.json()
        assert body["service"] == "MAGE Backend"

    def test_health_returns_version(self) -> None:
        response = client.get("/health")
        body = response.json()
        assert "version" in body
        assert body["version"] == "0.1.0"
