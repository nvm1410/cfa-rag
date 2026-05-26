"""Authentication: JWT tokens + register/login endpoints."""

import hashlib
import hmac
import json
import os
import time
import base64
from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.storage import get_user_by_username, create_user, get_user_by_id, User
from app.schemas import RegisterRequest, LoginRequest, AuthResponse

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET", "cfa-rag-dev-secret-change-in-prod")
TOKEN_TTL = 86400  # 24h


# ---------------------------------------------------------------------------
# JWT (HS256, zero dependencies)
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s)


def create_token(user_id: int, username: str) -> str:
    now = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(
        json.dumps({
            "user_id": user_id,
            "username": username,
            "iat": now,
            "exp": now + TOKEN_TTL,
        }).encode()
    )
    sig_input = f"{header}.{payload}"
    sig = _b64(
        hmac.new(SECRET_KEY.encode(), sig_input.encode(), hashlib.sha256).digest()
    )
    return f"{sig_input}.{sig}"


def decode_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        sig_input = f"{header_b64}.{payload_b64}"
        expected = _b64(
            hmac.new(SECRET_KEY.encode(), sig_input.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig_b64, expected):
            return None
        payload = json.loads(_unb64(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


# ---------------------------------------------------------------------------
# Dependency: get current user from Authorization header
# ---------------------------------------------------------------------------

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    payload = decode_token(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(payload["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=AuthResponse)
def register(body: RegisterRequest):
    username = body.username.strip().lower()
    if not username or len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(body.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    existing = get_user_by_username(username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    user = create_user(username, hash_password(body.password))
    token = create_token(user.id, user.username)
    return AuthResponse(token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    username = body.username.strip().lower()
    user = get_user_by_username(username)
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(user.id, user.username)
    return AuthResponse(token=token, user_id=user.id, username=user.username)
