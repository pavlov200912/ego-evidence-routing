"""Oracle router — stub for Week 3."""

from __future__ import annotations


class OracleRouter:
    """For each question, selects the tool that answered it correctly.

    Provides the upper-bound accuracy ceiling for evidence routing.
    To be implemented in Week 3.
    """

    def route(self, question_id: str, category: str) -> str:
        raise NotImplementedError("OracleRouter not yet implemented (Week 3).")
