"""
schemas/auth.py
───────────────
Pydantic v2 request / response models for authentication endpoints.

Includes strong validation:
  • Email format validation
  • Password complexity enforcement (min 8 chars, upper, lower, digit, special)
  • Explicit error messages for each constraint
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Request models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Payload for user registration."""

    email: EmailStr = Field(
        ...,
        description="User email address.",
        examples=["user@example.com"],
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="User's full name.",
        examples=["John Doe"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 chars, must include upper, lower, digit, special char).",
    )

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        """Enforce password complexity rules."""
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

    @field_validator("full_name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Strip whitespace and reject suspicious input."""
        v = v.strip()
        if "<" in v or ">" in v:
            raise ValueError("Name contains invalid characters.")
        return v


class LoginRequest(BaseModel):
    """Payload for user login."""

    email: EmailStr = Field(
        ...,
        description="Registered email address.",
        examples=["user@example.com"],
    )
    password: str = Field(
        ...,
        min_length=1,
        description="Account password.",
    )


class RefreshRequest(BaseModel):
    """Payload for token refresh."""

    refresh_token: str = Field(..., description="Valid refresh token.")


class UpdateProfileRequest(BaseModel):
    """Payload for updating the current user's profile. All fields optional —
    only provided fields are changed."""

    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=200,
        description="New full name.",
    )
    email: EmailStr | None = Field(
        default=None,
        description="New email address.",
    )
    default_expertise_level: str | None = Field(
        default=None,
        description="Default expertise level for new analyses.",
    )

    @field_validator("full_name")
    @classmethod
    def sanitize_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if "<" in v or ">" in v:
            raise ValueError("Name contains invalid characters.")
        return v

    @field_validator("default_expertise_level")
    @classmethod
    def validate_expertise_level(cls, v: str | None) -> str | None:
        if v is not None and v not in ("beginner", "intermediate", "expert"):
            raise ValueError("default_expertise_level must be one of: beginner, intermediate, expert.")
        return v


class ChangePasswordRequest(BaseModel):
    """Payload for changing the current user's password."""

    current_password: str = Field(..., min_length=1, description="Current password, for verification.")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password.")

    @field_validator("new_password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", v):
            raise ValueError("Password must contain at least one special character.")
        return v


# ── Response models ───────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    """JWT token pair returned after successful auth."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        ..., description="Access token lifetime in seconds."
    )


class UserResponse(BaseModel):
    """Public user profile returned by /me and /register."""

    id: str
    email: str
    full_name: str
    is_active: bool
    default_expertise_level: str
    created_at: datetime


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
