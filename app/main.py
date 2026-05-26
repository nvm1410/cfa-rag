"""FastAPI application — single-turn RAG + conversational chat API."""

from fastapi import FastAPI
from app.models import AskRequest as SingleTurnAskRequest
from app.rag import ask_rag
from app.auth import router as auth_router
from app.sessions import router as sessions_router
from app.chat import router as chat_router

app = FastAPI(title="CFA RAG API")

# Existing single-turn endpoint (unchanged)
@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: SingleTurnAskRequest):
    return ask_rag(req.question)


# New conversational endpoints
app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(chat_router)
