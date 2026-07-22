"""
tests/backend/test_auth.py
────────────────────────────
Integration tests for authentication and account-management endpoints.

Runs against the configured DATABASE_URL (Postgres) — there's no isolated
test database yet, so each test uses a unique, randomly-generated email
and deletes the user it creates (cascades to any datasets/runs) to avoid
polluting the database across runs.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _register_and_login(email: str, password: str = "Passw0rd!", full_name: str = "Test User") -> str:
    """Register a user and return an access token."""
    res = client.post(
        "/auth/register",
        json={"email": email, "full_name": full_name, "password": password},
    )
    assert res.status_code == 201, res.text
    res = client.post("/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


class TestRegisterAndLogin:
    def test_register_returns_user_profile(self) -> None:
        email = _unique_email()
        res = client.post(
            "/auth/register",
            json={"email": email, "full_name": "Alice", "password": "Passw0rd!"},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["email"] == email
        assert body["full_name"] == "Alice"
        assert body["default_expertise_level"] == "intermediate"
        assert "hashed_password" not in body

        token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
        client.delete("/auth/me", headers={"Authorization": f"Bearer {token}"})

    def test_register_duplicate_email_returns_409(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        res = client.post(
            "/auth/register",
            json={"email": email, "full_name": "Duplicate", "password": "Passw0rd!"},
        )
        assert res.status_code == 409
        client.delete("/auth/me", headers={"Authorization": f"Bearer {token}"})

    def test_register_weak_password_rejected(self) -> None:
        res = client.post(
            "/auth/register",
            json={"email": _unique_email(), "full_name": "Weak", "password": "weak"},
        )
        assert res.status_code == 422

    def test_login_wrong_password_returns_401(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        res = client.post("/auth/login", json={"email": email, "password": "WrongPass1!"})
        assert res.status_code == 401
        client.delete("/auth/me", headers={"Authorization": f"Bearer {token}"})

    def test_login_unknown_email_returns_401(self) -> None:
        res = client.post("/auth/login", json={"email": _unique_email(), "password": "Passw0rd!"})
        assert res.status_code == 401


class TestMe:
    def test_me_requires_auth(self) -> None:
        res = client.get("/auth/me")
        assert res.status_code == 401

    def test_me_returns_current_user(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json()["email"] == email
        client.delete("/auth/me", headers={"Authorization": f"Bearer {token}"})


class TestUpdateProfile:
    def test_update_full_name_and_expertise(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        headers = {"Authorization": f"Bearer {token}"}

        res = client.patch(
            "/auth/me",
            headers=headers,
            json={"full_name": "New Name", "default_expertise_level": "expert"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["full_name"] == "New Name"
        assert body["default_expertise_level"] == "expert"
        client.delete("/auth/me", headers=headers)

    def test_update_email_to_existing_returns_409(self) -> None:
        email_a = _unique_email()
        email_b = _unique_email()
        token_a = _register_and_login(email_a)
        token_b = _register_and_login(email_b)

        res = client.patch(
            "/auth/me",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"email": email_a},
        )
        assert res.status_code == 409

        client.delete("/auth/me", headers={"Authorization": f"Bearer {token_a}"})
        client.delete("/auth/me", headers={"Authorization": f"Bearer {token_b}"})

    def test_invalid_expertise_level_rejected(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        headers = {"Authorization": f"Bearer {token}"}

        res = client.patch("/auth/me", headers=headers, json={"default_expertise_level": "wizard"})
        assert res.status_code == 422
        client.delete("/auth/me", headers=headers)


class TestChangePassword:
    def test_wrong_current_password_returns_401(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        headers = {"Authorization": f"Bearer {token}"}

        res = client.post(
            "/auth/change-password",
            headers=headers,
            json={"current_password": "WrongOne1!", "new_password": "NewPassw0rd!"},
        )
        assert res.status_code == 401
        client.delete("/auth/me", headers=headers)

    def test_correct_current_password_updates_and_old_login_fails(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        headers = {"Authorization": f"Bearer {token}"}

        res = client.post(
            "/auth/change-password",
            headers=headers,
            json={"current_password": "Passw0rd!", "new_password": "NewPassw0rd!"},
        )
        assert res.status_code == 200

        old_login = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"})
        assert old_login.status_code == 401

        new_login = client.post("/auth/login", json={"email": email, "password": "NewPassw0rd!"})
        assert new_login.status_code == 200

        client.delete("/auth/me", headers={"Authorization": f"Bearer {new_login.json()['access_token']}"})


class TestDeleteAccount:
    def test_delete_then_login_fails(self) -> None:
        email = _unique_email()
        token = _register_and_login(email)
        headers = {"Authorization": f"Bearer {token}"}

        res = client.delete("/auth/me", headers=headers)
        assert res.status_code == 200

        login = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"})
        assert login.status_code == 401

    def test_delete_requires_auth(self) -> None:
        res = client.delete("/auth/me")
        assert res.status_code == 401
