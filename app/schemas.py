from pydantic import BaseModel
from typing import Optional


# ---- Auth ----

class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str


# ---- Sessions ----

class SessionCreateRequest(BaseModel):
    title: str = "New Chat"


class SessionResponse(BaseModel):
    id: str
    title: str
    summary: str
    created_at: str
    updated_at: str


class SessionDetailResponse(SessionResponse):
    messages: list[dict]


# ---- Chat ----

class AskRequest(BaseModel):
    question: str


class ChatAskResponse(BaseModel):
    answer: str
    session_id: str
    relevant: bool
    route: Optional[dict] = None
    contexts: Optional[list] = None
    multi_queries: Optional[list[str]] = None
    hyde_passage: Optional[str] = None
