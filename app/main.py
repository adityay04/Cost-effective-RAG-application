from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.service import ingest_file
from app.generation.answer import ask_question

app = FastAPI(
    title="Cost-Efficient RAG",
    description="A document QA system using ChromaDB and Groq",
    version="1.0.0"
)

class IngestRequest(BaseModel):
    file_path: str

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filename: str | None = None

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        return ingest_file(request.file_path)
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post("/ask")
def ask(request: AskRequest):
    metadata_filter = None

    if request.filename:
        metadata_filter = {
            "filename": request.filename
        }

    try:
        return ask_question(
            question=request.question,
            top_k=request.top_k,
            metadata_filter=metadata_filter
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
