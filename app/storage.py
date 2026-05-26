"""Database models and CRUD operations."""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Chat")
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


# ---- CRUD: Users ----

def get_user_by_username(username: str) -> Optional[User]:
    with SessionLocal() as db:
        return db.query(User).filter(User.username == username).first()


def get_user_by_id(user_id: int) -> Optional[User]:
    with SessionLocal() as db:
        return db.query(User).filter(User.id == user_id).first()


def create_user(username: str, password_hash: str) -> User:
    with SessionLocal() as db:
        user = User(username=username, password_hash=password_hash)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


# ---- CRUD: Sessions ----

def create_session(user_id: int, title: str = "New Chat") -> ChatSession:
    with SessionLocal() as db:
        session = ChatSession(user_id=user_id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session


def get_user_sessions(user_id: int) -> list[ChatSession]:
    with SessionLocal() as db:
        return (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )


def get_session(session_id: str) -> Optional[ChatSession]:
    with SessionLocal() as db:
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def update_session(session_id: str, **kwargs) -> bool:
    with SessionLocal() as db:
        result = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id)
            .update({**kwargs, "updated_at": datetime.now(timezone.utc)})
        )
        db.commit()
        return result > 0


def delete_session(session_id: str) -> bool:
    with SessionLocal() as db:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            return False
        db.query(Message).filter(Message.session_id == session_id).delete()
        db.delete(session)
        db.commit()
        return True


# ---- CRUD: Messages ----

def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
) -> Message:
    with SessionLocal() as db:
        msg = Message(
            session_id=session_id,
            role=role,
            content=content,
            metadata_json=json.dumps(metadata or {}),
        )
        db.add(msg)
        db.query(ChatSession).filter(ChatSession.id == session_id).update(
            {"updated_at": datetime.now(timezone.utc)}
        )
        db.commit()
        db.refresh(msg)
        return msg


def get_session_messages(session_id: str, limit: int = 50) -> list[Message]:
    with SessionLocal() as db:
        return (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .all()
        )


def count_session_messages(session_id: str) -> int:
    with SessionLocal() as db:
        return (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .count()
        )
