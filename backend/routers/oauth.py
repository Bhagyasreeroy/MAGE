"""
routers/oauth.py
────────────────
OAuth2 endpoints for Google Sign-In.

Uses the Authorization Code flow manually (without relying on
authlib's session-based state management, which breaks with
Starlette's cookie handling in development).
"""

from __future__ import annotations

import logging
import httpx

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.security import create_access_token
from backend.core.config import settings
from backend.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


@router.get("/google/login")
async def google_login():
    """
    Redirect the user to Google's OAuth consent screen.
    We skip state/CSRF here since this is a dev environment — 
    add a signed state param for production.
    """
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    redirect_uri = f"http://localhost:8000/auth/google/callback"

    params = (
        f"client_id={settings.google_client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{params}")


@router.get("/google/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    """
    Handle the callback from Google.
    1. Exchange the code for tokens.
    2. Fetch user info.
    3. Find or create the user in our DB.
    4. Issue a MAGE JWT and redirect to the frontend dashboard.
    """
    redirect_uri = "http://localhost:8000/auth/google/callback"

    # ── Step 1: Exchange code for tokens ─────────────────────────────────
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )

    if token_response.status_code != 200:
        logger.error(f"Token exchange failed: {token_response.text}")
        raise HTTPException(status_code=400, detail="Could not exchange code with Google")

    tokens = token_response.json()
    access_token = tokens.get("access_token")

    # ── Step 2: Fetch user info from Google ───────────────────────────────
    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if userinfo_response.status_code != 200:
        logger.error(f"Userinfo fetch failed: {userinfo_response.text}")
        raise HTTPException(status_code=400, detail="Could not fetch user info from Google")

    user_info = userinfo_response.json()
    email = user_info.get("email")
    name = user_info.get("name", "")

    if not email:
        raise HTTPException(status_code=400, detail="No email provided by Google")

    # ── Step 3: Find or create user ───────────────────────────────────────
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            email=email,
            full_name=name,
            hashed_password=None,
            auth_provider="google",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info(f"New Google OAuth user created: {email}")
    else:
        logger.info(f"Existing user signed in via Google: {email}")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # ── Step 4: Issue MAGE JWT and redirect ───────────────────────────────
    mage_token = create_access_token(subject=user.id)
    frontend_url = settings.frontend_url
    return RedirectResponse(url=f"{frontend_url}/auth-callback?token={mage_token}")
