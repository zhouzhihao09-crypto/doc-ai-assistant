from pathlib import Path
from collections import Counter
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "qwen3:4b"

SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 8
FINAL_TOP_K = 6

CHUNK_SIZE = 800


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    text = text.lower()

    replacements = {
        "licensed": "license",
        "licensing": "license",
        "licenses": "license",

        "requirements": "requirement",
        "prerequisites": "prerequisite",

        "clients": "client",
        "servers": "server",

        "configuring": "configure",
        "configuration": "configure",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def tokenize(text):

    text = normalize_text(text)

    english_tokens = re.findall(
        r"[a-zA-Z0-9_]+",
        text
    )

    chinese_tokens = re.findall(
        r"[\u4e00-\u9fff]",
        text
    )

    return english_tokens + chinese_tokens


# ============================================================
# PDF loading
# ============================================================

def load_pdf(pdf_file):

    pdf_file = Path(pdf_file)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}"
        )

    print(f"Reading PDF: {pdf_file.name}")

    reader = PdfReader(str(pdf_file))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    print(
        f"PDF read successfully! "
        f"{len(reader.pages)} pages found."
    )

    return pages


# ============================================================
# Chunking
# ============================================================

def create_chunks(pages):

    chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        paragraphs = re.split(
            r"\n\s*\n",
            text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Very large paragraph
            if len(paragraph) > CHUNK_SIZE:

                if current_chunk:

                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_number
                    })

                    current_chunk = ""

                for start in range(
                    0,
                    len(paragraph),
                    CHUNK_SIZE
                ):

                    piece = paragraph[
                        start:start + CHUNK_SIZE
                    ]

                    if piece.strip():

                        chunks.append({
                            "text": piece.strip(),
                            "page": page_number
                        })

                continue

            proposed = (
                current_chunk + "\n\n" + paragraph
                if current_chunk
                else paragraph
            )

            if len(proposed) <= CHUNK_SIZE:

                current_chunk = proposed

            else:

                if current_chunk:

                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_number
                    })

                current_chunk = paragraph

        if current_chunk:

            chunks.append({
                "text": current_chunk.strip(),
                "page": page_number
            })

    print(
        f"Document split into {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# Keyword index
# ============================================================

def build_keyword_index(chunks):

    print("Creating keyword index...")

    keyword_index = []

    for chunk in chunks:

        tokens = tokenize(
            chunk["text"]
        )

        keyword_index.append(
            Counter(tokens)
        )

    print("Keyword index created!")

    return keyword_index


# ============================================================
# Keyword search
# ============================================================

def keyword_search(
    question,
    chunks,
    keyword_index,
    top_k
):

    query_tokens = tokenize(
        question
    )

    if not query_tokens:
        return []

    query_counter = Counter(
        query_tokens
    )

    results = []

    for index, chunk_counter in enumerate(
        keyword_index
    ):

        score = 0

        for token, query_count in (
            query_counter.items()
        ):

            if token in chunk_counter:

                score += min(
                    query_count,
                    chunk_counter[token]
                )

        if score > 0:

            results.append(
                (index, score)
            )

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# RAG Engine
# ============================================================

class RAGEngine:

    def __init__(
        self,
        pdf_file,
        embedding_model=None
    ):

        print()
        print("=" * 60)
        print("Loading RAG engine...")
        print("=" * 60)

        self.pdf_file = Path(
            pdf_file
        )

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        pages = load_pdf(
            self.pdf_file
        )

        self.chunks = create_chunks(
            pages
        )

        # ----------------------------------------------------
        # Keyword index
        # ----------------------------------------------------

        self.keyword_index = (
            build_keyword_index(
                self.chunks
            )
        )

        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        if embedding_model is None:

            print(
                "Loading AI embedding model..."
            )

            self.embedding_model = (
                SentenceTransformer(
                    EMBEDDING_MODEL
                )
            )

        else:

            self.embedding_model = (
                embedding_model
            )

        print(
            "AI embedding model loaded!"
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        print(
            "Creating vector embeddings..."
        )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = (
            self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        )

        embeddings = embeddings.astype(
            "float32"
        )

        print(
            "Embeddings created!"
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings
        )

        print(
            "FAISS index created!"
        )

        print(
            f"RAG engine ready for: "
            f"{self.pdf_file.name}"
        )

        print("=" * 60)
        print()

    # ========================================================
    # Semantic search
    # ========================================================

    def semantic_search(
        self,
        question
    ):

        question_embedding = (
            self.embedding_model.encode(
                [question],
                convert_to_numpy=True
            ).astype("float32")
        )

        distances, indices = (
            self.index.search(
                question_embedding,
                SEMANTIC_TOP_K
            )
        )

        results = []

        for rank, index in enumerate(
            indices[0]
        ):

            if index < 0:
                continue

            results.append({
                "index": int(index),
                "rank": rank + 1,
                "distance": float(
                    distances[0][rank]
                )
            })

        return results

    # ========================================================
    # Hybrid search
    # ========================================================

    def hybrid_search(
        self,
        question
    ):

        semantic_results = (
            self.semantic_search(
                question
            )
        )

        keyword_results = (
            keyword_search(
                question,
                self.chunks,
                self.keyword_index,
                KEYWORD_TOP_K
            )
        )

        combined_scores = {}

        # ----------------------------------------------------
        # Semantic results
        # ----------------------------------------------------

        for result in semantic_results:

            index = result["index"]
            rank = result["rank"]

            score = (
                SEMANTIC_TOP_K
                - rank
                + 1
            )

            combined_scores[index] = (
                combined_scores.get(
                    index,
                    0
                )
                + score
            )

        # ----------------------------------------------------
        # Keyword results
        # ----------------------------------------------------

        for rank, (
            index,
            keyword_score
        ) in enumerate(
            keyword_results,
            start=1
        ):

            score = (
                KEYWORD_TOP_K
                - rank
                + 1
            ) * 1.5

            combined_scores[index] = (
                combined_scores.get(
                    index,
                    0
                )
                + score
            )

        # ----------------------------------------------------
        # Final ranking
        # ----------------------------------------------------

        ranked = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        final_results = []

        for index, score in ranked[
            :FINAL_TOP_K
        ]:

            chunk = self.chunks[
                index
            ]

            final_results.append({

                "index": index,

                "score": score,

                "text": chunk["text"],

                "page": chunk["page"]

            })

        return final_results

    # ========================================================
    # Answer question
    # ========================================================

    def answer_question(
        self,
        question
    ):

        question = question.strip()

        if not question:

            return {
                "answer": (
                    "Please enter a question."
                ),
                "sources": []
            }

        print()
        print("-" * 60)
        print(
            f"Question: {question}"
        )
        print("-" * 60)

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        results = self.hybrid_search(
            question
        )

        if not results:

            return {
                "answer": (
                    "I could not find "
                    "relevant information "
                    "in the uploaded document."
                ),
                "sources": []
            }

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context_parts = []

        for result in results:

            context_parts.append(
                f"[Page {result['page']}]\n"
                f"{result['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        system_prompt = """
You are a document question-answering assistant.

You must answer the user's question ONLY using
the information provided in the uploaded document.

Important rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the document does not contain enough information
   to answer the question, clearly say that the uploaded
   document does not provide enough information.
4. Give a direct and useful answer.
5. When appropriate, organize the answer using bullet points
   or numbered steps.
6. Preserve technical names, commands, configuration values,
   and terminology from the document.
"""

        user_prompt = f"""
Uploaded document context:

{context}

User question:

{question}

Answer the question using only the uploaded document context.
"""

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        try:

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            answer = (
                response["message"]["content"]
                .strip()
            )

        except Exception as e:

            answer = (
                "There was a problem "
                "communicating with Ollama: "
                f"{e}"
            )

        # ----------------------------------------------------
        # Source pages
        # ----------------------------------------------------

        source_pages = sorted(
            set(
                result["page"]
                for result in results
            )
        )

        return {
            "answer": answer,
            "sources": source_pages
        }


# ============================================================
# Standalone testing
# ============================================================

if __name__ == "__main__":

    pdf_path = input(
        "Enter the PDF path: "
    ).strip()

    engine = RAGEngine(
        pdf_path
    )

    while True:

        question = input(
            "\nAsk a question "
            "(or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        result = engine.answer_question(
            question
        )

        print("\nAnswer:")
        print(result["answer"])

        print(
            "\nSource pages:",
            result["sources"]
        )