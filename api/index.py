import os
import tempfile
from pathlib import Path
from typing import Any

import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from psycopg.types.json import Json


POSTGRES_URL = os.getenv("POSTGRES_URL", "")
PGVECTOR_URL = os.getenv(
    "PGVECTOR_URL",
    POSTGRES_URL.replace("postgresql://", "postgresql+psycopg://", 1),
)
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "rag_chat_vercel")

app = FastAPI(title="RAG Chat API", version="1.0.0")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(min_length=1, max_length=12000)
    temperature: float = Field(default=0.3, ge=0, le=1)


def require_configuration() -> None:
    missing = [
        name
        for name, value in (
            ("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY")),
            ("POSTGRES_URL", POSTGRES_URL),
            ("PGVECTOR_URL", PGVECTOR_URL),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def db_connection() -> psycopg.Connection:
    require_configuration()
    return psycopg.connect(POSTGRES_URL)


def initialize_database() -> None:
    with db_connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                title TEXT DEFAULT 'Nouvelle conversation',
                pdf_files JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id BIGINT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                temperature DOUBLE PRECISION DEFAULT 0.5,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def vector_store() -> PGVector:
    require_configuration()
    return PGVector(
        collection_name=VECTOR_COLLECTION,
        embedding_function=MistralAIEmbeddings(
            model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
        ),
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )


def create_conversation(title: str, files: list[str]) -> int:
    with db_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO conversations (title, pdf_files)
            VALUES (%s, %s)
            RETURNING id
            """,
            (title, Json(files)),
        ).fetchone()
        if row is None:
            raise RuntimeError("Conversation creation returned no identifier")
        return int(row[0])


def conversation_exists(conversation_id: int) -> bool:
    with db_connection() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s", (conversation_id,)
            ).fetchone()
            is not None
        )


def save_message(
    conversation_id: int, role: str, content: str, temperature: float = 0.3
) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, temperature)
            VALUES (%s, %s, %s, %s)
            """,
            (conversation_id, role, content, temperature),
        )


def conversation_history(conversation_id: int) -> list[dict[str, str]]:
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at, id
            """,
            (conversation_id,),
        ).fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def answer_question(request: ChatRequest) -> str:
    context_documents = vector_store().similarity_search(
        request.message,
        k=6,
        filter={"conversation_id": {"$eq": str(request.conversation_id)}},
    )
    context = "\n\n".join(document.page_content for document in context_documents)
    prompt = (
        "Réponds en français, uniquement à partir du contexte fourni. "
        "Si le contexte ne contient pas la réponse, dis que tu ne disposes pas "
        "de cette information. Réponds en texte simple, sans Markdown.\n\n"
        f"Contexte :\n{context}\n\nQuestion :\n{request.message}"
    )
    response = ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=request.temperature,
        max_retries=2,
    ).invoke(
        [
            SystemMessage(content="Tu es un assistant RAG précis et utile."),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content)


@app.get("/health")
def health() -> dict[str, str]:
    require_configuration()
    initialize_database()
    return {"status": "ok"}


app.add_api_route("/api/health", health, methods=["GET"])


@app.get("/conversations")
def list_conversations() -> list[dict[str, Any]]:
    initialize_database()
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at
            FROM conversations
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [{"id": row[0], "title": row[1], "created_at": row[2]} for row in rows]


app.add_api_route("/api/conversations", list_conversations, methods=["GET"])


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int) -> dict[str, Any]:
    initialize_database()
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    return {
        "id": conversation_id,
        "messages": conversation_history(conversation_id),
    }


app.add_api_route(
    "/api/conversations/{conversation_id}", get_conversation, methods=["GET"]
)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    initialize_database()
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Un fichier PDF est requis")

    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await file.read(1024 * 1024):
                temporary_file.write(chunk)

        documents = PyPDFLoader(temporary_path).load()
        conversation_id = create_conversation(
            f"Conversation - {file.filename or 'document.pdf'}",
            [file.filename or "document.pdf"],
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["conversation_id"] = str(conversation_id)
        vector_store().add_documents(chunks)
        return {"id": conversation_id, "filename": file.filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import PDF impossible: {exc}") from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


app.add_api_route("/api/upload", upload_pdf, methods=["POST"])


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    initialize_database()
    if not conversation_exists(request.conversation_id):
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    save_message(request.conversation_id, "user", request.message, request.temperature)
    try:
        response = answer_question(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Réponse Mistral impossible: {exc}") from exc
    save_message(request.conversation_id, "assistant", response, request.temperature)
    return {"answer": response}


app.add_api_route("/api/chat", chat, methods=["POST"])


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int) -> dict[str, bool]:
    initialize_database()
    with db_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = %s", (conversation_id,)
        )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation introuvable")
    return {"deleted": True}


app.add_api_route(
    "/api/conversations/{conversation_id}",
    delete_conversation,
    methods=["DELETE"],
)
