"""
routers/oauth.py
────────────────
OAuth2 endpoints for Google Sign-In.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.security import create_access_token, create_refresh_token
from backend.core.config import settings
from backend.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()
oauth = OAuth()

if settings.google_client_id and settings.google_client_secret:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={
            "scope": "openid email profile",
        },
    )

@router.get("/google/login")
async def google_login(request: Request):
    """
    Redirect the user to Google's consent screen.
    """
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/google/callback")
async def google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Handle the callback from Google.
    Creates a user if they don't exist, logs them in, and redirects to the frontend with a token.
    """
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
    except Exception as e:
        logger.error(f"OAuth error: {e}")
        raise HTTPException(status_code=400, detail="Could not validate Google credentials")

    if not user_info or not user_info.get("email"):
        raise HTTPException(status_code=400, detail="No email provided by Google")

    email = user_info["email"]
    name = user_info.get("name", "")

    # Look for existing user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        # Create new OAuth user (no password)
        user = User(
            email=email,
            full_name=name,
            hashed_password=None,
            auth_provider="google"
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info(f"New Google OAuth user registered: {email}")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Generate JWTs
    access_token = create_access_token(subject=user.id)
    
    # Redirect back to the frontend with the token (for simplicity in development)
    # In production, setting an HttpOnly cookie is preferred.
    redirect_url = f"{settings.frontend_url}/dashboard?token={access_token}"
    return RedirectResponse(url=redirect_url)
