"""Session management endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user
from app.storage import (
    User,
    create_session,
    get_user_sessions,
    get_session,
    update_session,
    delete_session,
    get_session_messages,
)
from app.schemas import SessionCreateRequest, SessionResponse, SessionDetailResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
def create(user: User = Depends(get_current_user), body: SessionCreateRequest = None):
    """Create a new chat session."""
    title = body.title if body else "New Chat"
    session = create_session(user.id, title=title)
    return SessionResponse(
        id=session.id,
        title=session.title,
        summary=session.summary,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.get("", response_model=list[SessionResponse])
def list_sessions(user: User = Depends(get_current_user)):
    """List all sessions for the current user (most recent first)."""
    sessions = get_user_sessions(user.id)
    return [
        SessionResponse(
            id=s.id,
            title=s.title,
            summary=s.summary,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get(session_id: str, user: User = Depends(get_current_user)):
    """Get a session with its message history."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your session")

    messages = get_session_messages(session_id)
    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        summary=session.summary,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        messages=[
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    )


@router.delete("/{session_id}", status_code=204)
def delete(session_id: str, user: User = Depends(get_current_user)):
    """Delete a session and all its messages."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your session")

    delete_session(session_id)


@router.patch("/{session_id}/title", response_model=SessionResponse)
def update_title(
    session_id: str,
    title: str,
    user: User = Depends(get_current_user),
):
    """Update session title."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your session")

    update_session(session_id, title=title)
    session = get_session(session_id)
    return SessionResponse(
        id=session.id,
        title=session.title,
        summary=session.summary,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )
