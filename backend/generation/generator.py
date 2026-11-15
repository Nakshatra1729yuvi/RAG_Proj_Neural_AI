import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from backend.ingestion.chunker import DocumentChunk
from backend.generation.safety import SafetyGuard
from backend.generation.prompt_templates import SYSTEM_PROMPT, build_rag_prompt
from backend.generation.llm_client import OllamaClient
from backend.generation.citation import CitationVerifier, CitationVerificationReport

class GenerationResult(BaseModel):
    answer: str
    raw_answer: str
    is_emergency: bool = False
    is_supported: bool = True
    citation_report: CitationVerificationReport
    follow_up_questions: List[str] = Field(default_factory=list)
    used_chunks: List[DocumentChunk] = Field(default_factory=list)

class AnswerGenerator:
    """
    Orchestrates grounded healthcare answer generation with safety guardrails,
    strict citation auditing, and follow-up question generation.
    """

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[DocumentChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        min_relevance_passed: bool = True
    ) -> GenerationResult:
        # 1. Check for emergency
        is_emergency, emergency_msg = SafetyGuard.check_emergency(query)

        # 2. Check if unsupported / no chunks
        if not retrieved_chunks or not min_relevance_passed:
            unsupported_text = SafetyGuard.get_unsupported_response()
            if is_emergency:
                unsupported_text = f"{emergency_msg}\n\n---\n\n{unsupported_text}"
            
            return GenerationResult(
                answer=unsupported_text,
                raw_answer=unsupported_text,
                is_emergency=is_emergency,
                is_supported=False,
                citation_report=CitationVerificationReport(),
                follow_up_questions=[
                    "What medical conditions are covered in the loaded documents?",
                    "Can you summarize the general health guidelines available?",
                    "How should I consult a doctor regarding these symptoms?"
                ],
                used_chunks=[]
            )

        # 3. Build grounded prompt
        prompt = build_rag_prompt(
            query=query,
            chunks=retrieved_chunks,
            conversation_history=conversation_history
        )

        # 4. Generate LLM response
        raw_response = self.llm_client.generate(prompt=prompt, system=SYSTEM_PROMPT)

        # 5. Detect if LLM self-identified unsupported question
        lower_resp = raw_response.lower()
        unsupported_markers = [
            "insufficient evidence",
            "not mentioned in the provided",
            "does not provide information",
            "cannot answer based on the reference",
            "no information is available"
        ]
        is_supported = not any(marker in lower_resp for marker in unsupported_markers)

        # 6. Verify citations
        citation_report = CitationVerifier.verify(raw_response, retrieved_chunks)

        # If LLM didn't cite properly but answered factually from top chunk, auto-tag primary source
        augmented_response = raw_response
        if is_supported and citation_report.total_citations == 0 and retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            augmented_response += f"\n\n*(Primary Evidence Source: [Doc: {top_chunk.filename}, p.{top_chunk.page_number}])* "
            # Re-verify
            citation_report = CitationVerifier.verify(augmented_response, retrieved_chunks)

        # 7. Generate follow-up questions
        follow_ups = self._generate_follow_ups(query, raw_response, retrieved_chunks)

        # 8. Append healthcare disclaimer
        final_answer = SafetyGuard.append_disclaimer(augmented_response, is_emergency=is_emergency)

        return GenerationResult(
            answer=final_answer,
            raw_answer=raw_response,
            is_emergency=is_emergency,
            is_supported=is_supported,
            citation_report=citation_report,
            follow_up_questions=follow_ups,
            used_chunks=retrieved_chunks
        )

    def _generate_follow_ups(
        self,
        query: str,
        answer: str,
        chunks: List[DocumentChunk]
    ) -> List[str]:
        # Generate 2-3 contextual follow-up questions based on retrieved section headings and content
        topics = set()
        for c in chunks:
            if c.section_heading and len(c.section_heading) > 4:
                topics.add(c.section_heading)

        follow_ups = []
        for t in list(topics)[:2]:
            follow_ups.append(f"What additional details are available regarding {t}?")

        if not follow_ups:
            follow_ups = [
                "What are the recommended prevention strategies for this condition?",
                "What diagnostic criteria or screening tests are mentioned?",
                "When should someone consult a physician for these symptoms?"
            ]
        elif len(follow_ups) < 3:
            follow_ups.append("When should someone seek medical consultation for these symptoms?")

        return follow_ups[:3]
