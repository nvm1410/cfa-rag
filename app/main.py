from fastapi import FastAPI
from app.models import AskRequest
from app.rag import ask_rag

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    return ask_rag(req.question)