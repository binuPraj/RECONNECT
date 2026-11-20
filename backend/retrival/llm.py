"""OpenRouter wrapper that always writes parser/answer calls to the debug log."""
from __future__ import annotations

from retrival.debug_log import log_llm


class RetrievalLLM:
    def __init__(self, client=None) -> None:
        self.client = client
        if client is None:
            try:
                from app.memory.openrouter import OpenRouterMemoryClient
                self.client = OpenRouterMemoryClient()
            except Exception:
                self.client = None

    def complete(self, prompt: str, *, kind: str, patient_id: int | None, session_id: str | None,
                 question: str | None, extra: dict | None = None) -> str | None:
        if self.client is None:
            log_llm(kind=kind, prompt=prompt, response=None, patient_id=patient_id,
                    session_id=session_id, question=question, extra={"error": "llm_unavailable", **(extra or {})})
            return None
        try:
            response = self.client(prompt)
        except Exception as error:
            log_llm(kind=kind, prompt=prompt, response=None, patient_id=patient_id,
                    session_id=session_id, question=question, extra={"error": str(error), **(extra or {})})
            return None
        log_llm(kind=kind, prompt=prompt, response=response, patient_id=patient_id,
                session_id=session_id, question=question, extra=extra)
        return response
