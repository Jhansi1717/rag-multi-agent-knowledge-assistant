"""ConversationMemoryAgent — lightweight recent-turn memory."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from agents.models import ConversationTurn


class ConversationMemoryAgent:
    """Store and retrieve the last N turns per session."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._sessions: Dict[str, Deque[ConversationTurn]] = defaultdict(deque)

    def add_turn(self, session_id: str, query: str, answer: str) -> ConversationTurn:
        turns = self._sessions[session_id]
        turn = ConversationTurn(
            session_id=session_id,
            query=query,
            answer=answer,
            turn_index=len(turns),
        )
        turns.append(turn)
        while len(turns) > self.max_turns:
            turns.popleft()
        return turn

    def get_recent_turns(self, session_id: str, limit: int = 3) -> List[ConversationTurn]:
        turns = list(self._sessions.get(session_id, []))
        return turns[-limit:]

    def get_recent_context(self, session_id: str) -> str:
        turns = self.get_recent_turns(session_id, limit=1)
        if not turns:
            return ""
        last = turns[-1]
        return f"Previous question: {last.query} Previous answer: {last.answer}"

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
