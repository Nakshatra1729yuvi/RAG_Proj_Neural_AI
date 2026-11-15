from typing import List, Dict, Any
from backend.ingestion.chunker import DocumentChunk

SYSTEM_PROMPT = """You are MediRAG, an autonomous healthcare information assistant built to provide evidence-backed, factual educational health information.

CRITICAL INSTRUCTIONS:
1. Rely EXCLUSIVELY on the provided Reference Documents below.
2. If the reference documents DO NOT contain sufficient information to answer the question, state:
   "Based on the provided medical reference documents, there is insufficient evidence to answer your question."
   DO NOT fabricate, extrapolate, or guess.
3. Every factual statement or clinical claim MUST be accompanied by an inline citation referencing the exact document and page number in the format:
   [Doc: <filename>, p.<page_number>]
4. Maintain a professional, objective tone. Do NOT provide personal medical diagnoses or individualized prescriptions.
5. Emphasize that the user should consult licensed medical professionals for individual diagnosis and treatment.
"""

def format_context_passages(chunks: List[DocumentChunk]) -> str:
    if not chunks:
        return "No reference documents available."

    formatted_passages = []
    for idx, c in enumerate(chunks, start=1):
        heading_info = f" | Section: {c.section_heading}" if c.section_heading else ""
        header = f"--- Document #{idx}: {c.filename} (Page {c.page_number}{heading_info}) ---"
        formatted_passages.append(f"{header}\n{c.text}\n")

    return "\n".join(formatted_passages)

def build_rag_prompt(
    query: str,
    chunks: List[DocumentChunk],
    conversation_history: List[Dict[str, str]] = None
) -> str:
    context = format_context_passages(chunks)

    history_str = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-4:]:  # keep last 4 turns for context
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {turn.get('content', '')}")
        history_str = "Prior Conversation History:\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""{history_str}Reference Documents:
{context}

User Question: {query}

Provide a comprehensive, evidence-grounded answer based strictly on the Reference Documents above. Cite every key claim using [Doc: <filename>, p.<page_number>]. If the information is not present in the Reference Documents, say so clearly."""

    return prompt
