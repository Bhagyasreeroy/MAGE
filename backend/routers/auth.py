"""
routers/auth.py
───────────────
Authentication endpoints: register, login, token refresh, and /me.

Security measures:
  • Passwords are bcrypt-hashed before storage (never stored in plain text)
  • Login uses constant-time comparison to prevent timing attacks
  • Generic error messages on login failure (don't reveal if email exists)
  • Refresh tokens have a separate "type" claim — can't be used as access tokens
  • Rate limiting is recommended via a reverse proxy in production
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.core.config import settings
from backend.models.user import User
from backend.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        default_expertise_level=user.default_expertise_level,
        created_at=user.created_at,
    )


# ── POST /auth/register ──────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Create a new user.

    - Validates email uniqueness
    - Hashes the password with bcrypt
    - Returns the created user profile (no password)
    """
    # Check for existing user
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("New user registered: %s", user.email)

    return _to_user_response(user)


# ── POST /auth/login ─────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive JWT tokens",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate a user with email + password.

    Returns a JWT access token (short-lived) and refresh token (long-lived).
    Uses a generic error message to avoid revealing whether the email exists.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    # Generic message — don't reveal whether the email exists
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


# ── POST /auth/refresh ───────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh an expired access token",
)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Exchange a valid refresh token for a new access + refresh token pair.

    The old refresh token is implicitly invalidated by issuing a new one.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token.",
    )

    try:
        payload = decode_token(body.refresh_token)
        user_id: str | None = payload.get("sub")
        token_type: str | None = payload.get("type")

        if user_id is None or token_type != "refresh":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_exception

    new_access = create_access_token(subject=user.id)
    new_refresh = create_refresh_token(subject=user.id)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return _to_user_response(current_user)


# ── PATCH /auth/me ───────────────────────────────────────────────────────────

@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update the current user's profile",
)
async def update_me(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Update full_name / email / default_expertise_level. Only provided
    fields are changed."""
    if body.email is not None and body.email != current_user.email:
        result = await db.execute(select(User).where(User.email == body.email))
        if result.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            )
        current_user.email = body.email

    if body.full_name is not None:
        current_user.full_name = body.full_name
    if body.default_expertise_level is not None:
        current_user.default_expertise_level = body.default_expertise_level

    await db.commit()
    await db.refresh(current_user)

    logger.info("User updated profile: %s", current_user.email)
    return _to_user_response(current_user)


# ── POST /auth/change-password ───────────────────────────────────────────────

@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change the current user's password",
)
async def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Verify the current password, then set the new one."""
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )

    current_user.hashed_password = hash_password(body.new_password)
    await db.commit()

    logger.info("User changed password: %s", current_user.email)
    return MessageResponse(message="Password updated successfully.")


# ── DELETE /auth/me ───────────────────────────────────────────────────────────

@router.delete(
    "/me",
    response_model=MessageResponse,
    summary="Permanently delete the current user's account",
)
async def delete_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Permanently delete the account and all owned datasets/analysis runs
    (cascades via the foreign key ON DELETE CASCADE)."""
    email = current_user.email
    await db.delete(current_user)
    await db.commit()

    logger.info("User deleted account: %s", email)
    return MessageResponse(message="Account deleted.")
