import os
import re
import math
import faiss

from collections import Counter

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ollama import chat


# ============================================================
# SETTINGS
# ============================================================

PDF_FILE = "company_policy.pdf"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

OLLAMA_MODEL = "qwen3:4b"

# Number of semantic results
SEMANTIC_TOP_K = 8

# Number of keyword results
KEYWORD_TOP_K = 8

# Number of final chunks sent to Qwen3
FINAL_TOP_K = 6

# Approximate maximum chunk size
CHUNK_SIZE = 800


# ============================================================
# 1. LOAD PDF
# ============================================================

def load_pdf(filename):

    reader = PdfReader(filename)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text and page_text.strip():

            pages.append({
                "page": page_number,
                "text": page_text.strip()
            })

    return pages


# ============================================================
# 2. SPLIT PDF INTO CHUNKS
# ============================================================

def split_text(pages, chunk_size=800):

    chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        paragraphs = text.split("\n")

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) <= chunk_size:

                if current_chunk:
                    current_chunk += "\n"

                current_chunk += paragraph

            else:

                if current_chunk:

                    chunks.append({
                        "text": current_chunk,
                        "page": page_number
                    })

                current_chunk = paragraph

        if current_chunk:

            chunks.append({
                "text": current_chunk,
                "page": page_number
            })

    return chunks


# ============================================================
# 3. TOKENIZE TEXT
# ============================================================

def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    return words


# ============================================================
# 4. NORMALIZE TECHNICAL WORDS
# ============================================================

def normalize_word(word):

    # Handle common variations found in technical manuals

    replacements = {

        "licensed": "license",
        "licensing": "license",
        "licenses": "license",

        "requirements": "requirement",

        "prerequisites": "prerequisite",

        "clients": "client",

        "servers": "server",

        "configuring": "configure",
        "configured": "configure",
        "configuration": "configure",
    }

    return replacements.get(
        word,
        word
    )


# ============================================================
# 5. PREPARE KEYWORDS
# ============================================================

def prepare_keywords(text):

    words = tokenize(text)

    return [
        normalize_word(word)
        for word in words
        if len(word) > 2
    ]


# ============================================================
# 6. BUILD KEYWORD INDEX
# ============================================================

def build_keyword_index(chunks):

    index = []

    for chunk in chunks:

        words = prepare_keywords(
            chunk["text"]
        )

        index.append(
            Counter(words)
        )

    return index


# ============================================================
# 7. KEYWORD SEARCH
# ============================================================

def keyword_search(
    question,
    chunks,
    keyword_index,
    top_k=8
):

    query_words = prepare_keywords(
        question
    )

    query_counts = Counter(
        query_words
    )

    scored_chunks = []

    for i, word_counts in enumerate(
        keyword_index
    ):

        score = 0

        for word, query_count in query_counts.items():

            if word in word_counts:

                # Exact keyword matches receive a strong score

                score += (
                    1
                    + math.log1p(
                        word_counts[word]
                    )
                )

        if score > 0:

            scored_chunks.append(
                (score, i)
            )

    scored_chunks.sort(
        reverse=True
    )

    return scored_chunks[:top_k]


# ============================================================
# 8. CHECK PDF
# ============================================================

if not os.path.exists(PDF_FILE):

    print(
        f"\nERROR: Cannot find {PDF_FILE}"
    )

    print(
        "Make sure company_policy.pdf and "
        "main.py are in the same folder."
    )

    exit()


# ============================================================
# 9. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded!")


# ============================================================
# 10. READ PDF
# ============================================================

print(
    "\nReading NVIDIA AI Enterprise User Guide..."
)

pages = load_pdf(
    PDF_FILE
)

if not pages:

    print(
        "\nERROR: No text was found in the PDF."
    )

    exit()


print(
    f"PDF read successfully! "
    f"Found {len(pages)} pages with text."
)


# ============================================================
# 11. CREATE CHUNKS
# ============================================================

chunks = split_text(
    pages,
    chunk_size=CHUNK_SIZE
)

print(
    f"Document split into {len(chunks)} "
    f"paragraph-based chunks."
)


# ============================================================
# 12. BUILD KEYWORD INDEX
# ============================================================

print(
    "\nBuilding keyword search index..."
)

keyword_index = build_keyword_index(
    chunks
)

print(
    "Keyword search index created!"
)


# ============================================================
# 13. CREATE EMBEDDINGS
# ============================================================

print(
    "\nCreating vector embeddings..."
)

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype(
    "float32"
)

print(
    "Embeddings created!"
)


# ============================================================
# 14. CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

print(
    "FAISS index created!"
)


# ============================================================
# 15. CHECK OLLAMA
# ============================================================

print(
    "\nChecking Ollama..."
)

try:

    chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Reply with the word READY."
            }
        ]
    )

    print(
        "Ollama connection successful!"
    )

except Exception as e:

    print(
        "\nERROR: Could not connect to Ollama."
    )

    print(
        "Make sure Ollama is installed and running."
    )

    print("\nDetails:")

    print(e)

    exit()


# ============================================================
# 16. START ASSISTANT
# ============================================================

print(
    "\n=============================================="
)

print(
    " NVIDIA AI Enterprise RAG Assistant"
)

print(
    "=============================================="
)

print(
    "\nHybrid search enabled:"
)

print(
    "  • Semantic search"
)

print(
    "  • Keyword search"
)

print(
    "  • Combined retrieval"
)

print(
    "\nYour PDF is connected to Qwen3."
)

print(
    "\nAsk questions about the"
)

print(
    "NVIDIA AI Enterprise User Guide."
)

print(
    "\nType q to quit."
)


# ============================================================
# 17. QUESTION LOOP
# ============================================================

while True:

    question = input(
        "\nAsk a question: "
    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if question.lower() == "q":

        print(
            "\nProgram ended."
        )

        break


    # --------------------------------------------------------
    # Ignore empty questions
    # --------------------------------------------------------

    if not question.strip():

        continue


    # ========================================================
    # 18. SEMANTIC SEARCH
    # ========================================================

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(
        "float32"
    )

    semantic_distances, semantic_indices = index.search(
        question_embedding,
        SEMANTIC_TOP_K
    )


    # ========================================================
    # 19. KEYWORD SEARCH
    # ========================================================

    keyword_results = keyword_search(
        question,
        chunks,
        keyword_index,
        KEYWORD_TOP_K
    )


    # ========================================================
    # 20. COMBINE SEARCH RESULTS
    # ========================================================

    combined_scores = {}

    # --------------------------------------------------------
    # Semantic ranking
    # --------------------------------------------------------

    for rank, idx in enumerate(
        semantic_indices[0],
        start=1
    ):

        # Higher rank = higher score

        score = (
            SEMANTIC_TOP_K - rank + 1
        )

        combined_scores[idx] = (
            combined_scores.get(idx, 0)
            + score
        )


    # --------------------------------------------------------
    # Keyword ranking
    # --------------------------------------------------------

    for rank, (keyword_score, idx) in enumerate(
        keyword_results,
        start=1
    ):

        # Keyword matches get strong weight

        score = (
            KEYWORD_TOP_K - rank + 1
        ) * 1.5

        combined_scores[idx] = (
            combined_scores.get(idx, 0)
            + score
        )


    # ========================================================
    # 21. SORT COMBINED RESULTS
    # ========================================================

    ranked_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # Keep only the best results

    final_indices = [
        idx
        for idx, score
        in ranked_results[:FINAL_TOP_K]
    ]


    # ========================================================
    # 22. BUILD DOCUMENT CONTEXT
    # ========================================================

    context_parts = []

    for rank, idx in enumerate(
        final_indices,
        start=1
    ):

        result = chunks[idx]

        context_parts.append(
            f"[Source {rank} - Page {result['page']}]\n"
            f"{result['text']}"
        )


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # 23. ASK QWEN3
    # ========================================================

    system_prompt = """
You are a technical documentation assistant.

Your job is to answer questions about the
NVIDIA AI Enterprise User Guide.

IMPORTANT RULES:

1. Use ONLY the documentation context provided.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Prefer information that directly answers the question.
5. If multiple sources contain relevant information,
   combine them into one clear answer.
6. When possible, mention the relevant page number.
7. If the documentation does not contain enough
   information, clearly say so.
8. Keep the answer concise and useful.

The source material is the NVIDIA AI Enterprise
User Guide.
"""


    user_prompt = f"""
User question:

{question}


Documentation context:

{context}
"""


    print(
        "\nThinking...\n"
    )


    try:

        response = chat(

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

        answer = response.message.content


    except Exception as e:

        print(
            "\nERROR while asking Qwen3:"
        )

        print(e)

        continue


    # ========================================================
    # 24. DISPLAY ANSWER
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        " Answer"
    )

    print(
        "==============================================\n"
    )

    print(
        answer
    )


    # ========================================================
    # 25. DISPLAY SOURCES
    # ========================================================

    source_pages = sorted(
        set(
            chunks[idx]["page"]
            for idx in final_indices
        )
    )

    print(
        "\nSource pages: "
        + ", ".join(
            str(page)
            for page in source_pages
        )
    )

    print(
        "\n=============================================="
    )