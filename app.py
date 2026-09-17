from pathlib import Path
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Cookie,
    HTTPException
)

from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from rag_engine import RAGEngine


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

HTML_FILE = (
    BASE_DIR
    / "templates"
    / "index.html"
)

UPLOAD_DIR = (
    BASE_DIR
    / "uploads"
)

UPLOAD_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Document AI Assistant"
)


# ============================================================
# Shared embedding model
# ============================================================

print()
print("=" * 60)
print("Loading shared embedding model...")
print("=" * 60)

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Shared embedding model loaded!")
print("=" * 60)
print()


# ============================================================
# User sessions
# ============================================================

# Each browser receives a unique session ID.
#
# For this first version, each session has its own RAG engine.
#
# Example:
#
# Phone A → PDF A → RAG engine A
# Laptop  → PDF B → RAG engine B
#
# This prevents one user's uploaded PDF from replacing
# another user's document.

user_engines = {}

user_documents = {}


# ============================================================
# Request model
# ============================================================

class QuestionRequest(BaseModel):

    question: str


# ============================================================
# Get or create session
# ============================================================

def get_session_id(
    session_id
):

    if session_id:
        return session_id

    return str(
        uuid.uuid4()
    )


# ============================================================
# Website
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    if not HTML_FILE.exists():

        return HTMLResponse(
            content=(
                "<h1>Error</h1>"
                "<p>templates/index.html "
                "was not found.</p>"
            ),
            status_code=500
        )

    return HTML_FILE.read_text(
        encoding="utf-8"
    )


# ============================================================
# Upload PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str | None = Cookie(
        default=None
    )
):

    session_id = get_session_id(
        session_id
    )

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    filename = file.filename

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    # --------------------------------------------------------
    # Create session folder
    # --------------------------------------------------------

    session_dir = (
        UPLOAD_DIR
        / session_id
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = (
        session_dir
        / filename
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    with pdf_path.open(
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    print()
    print("=" * 60)
    print(
        f"New PDF uploaded: {filename}"
    )
    print(
        f"Session: {session_id}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Build RAG engine
    # --------------------------------------------------------

    try:

        engine = RAGEngine(
            pdf_path,
            embedding_model=embedding_model
        )

    except Exception as e:

        print(
            f"Error processing PDF: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not process the PDF. "
                f"Error: {e}"
            )
        )

    # --------------------------------------------------------
    # Store engine for this browser
    # --------------------------------------------------------

    user_engines[
        session_id
    ] = engine

    user_documents[
        session_id
    ] = filename

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = {
        "filename": filename,
        "message": (
            "PDF uploaded and processed successfully."
        )
    }

    from fastapi.responses import JSONResponse

    json_response = JSONResponse(
        content=response
    )

    json_response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax"
    )

    return json_response


# ============================================================
# Ask question
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    session_id: str | None = Cookie(
        default=None
    )
):

    if not session_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a PDF first."
            )
        )

    engine = user_engines.get(
        session_id
    )

    if engine is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a PDF first."
            )
        )

    result = engine.answer_question(
        request.question
    )

    result["filename"] = (
        user_documents.get(
            session_id
        )
    )

    return result