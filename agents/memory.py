"""Bounded Conversation Memory Service."""

from __future__ import annotations

import time
import re
from collections import defaultdict
from typing import Dict, List, Optional

from agents.models import ConversationTurn, ClarificationState
from agents.query_understanding import QueryUnderstandingAgent


class ConversationMemoryAgent:
    """Store and retrieve bounded session history with relevance selection."""

    def __init__(
        self,
        max_turns: int = 10,
        max_characters: int = 4000,
        max_clarifications: int = 3,
        max_sessions: int = 1000,
        qu_agent: Optional[QueryUnderstandingAgent] = None,
    ):
        self.max_turns = max_turns
        self.max_characters = max_characters
        self.max_clarifications = max_clarifications
        self.max_sessions = max_sessions
        self.qu = qu_agent or QueryUnderstandingAgent()

        self._sessions: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self._clarifications: Dict[str, ClarificationState] = {}
        self._clarification_counts: Dict[str, int] = defaultdict(int)

    def create_session(self, session_id: str) -> None:
        """Initialize empty session structures."""
        if session_id not in self._sessions:
            if len(self._sessions) >= self.max_sessions:
                raise ValueError("Maximum active sessions reached.")
            self._sessions[session_id] = []
            self._clarification_counts[session_id] = 0

    def get_session(self, session_id: str) -> List[ConversationTurn]:
        """Validate and return session history."""
        if session_id not in self._sessions:
            raise ValueError(f"Unknown session: {session_id}")
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Purge all data for a session."""
        if session_id not in self._sessions:
            raise ValueError(f"Unknown session: {session_id}")
        self._sessions.pop(session_id, None)
        self._clarifications.pop(session_id, None)
        self._clarification_counts.pop(session_id, None)

    def list_session_history(self, session_id: str) -> List[ConversationTurn]:
        """Return the full bounded turn list for a session."""
        return self.get_session(session_id)

    def append_turn(self, session_id: str, turn: ConversationTurn) -> ConversationTurn:
        """Append turn, enforce limits, and clear resolved clarifications."""
        self.create_session(session_id)
        turns = self._sessions[session_id]

        # Enforce turn bounds before append
        if len(turns) >= self.max_turns:
            turns.pop(0)

        turn.turn_index = len(turns)
        turns.append(turn)

        # Reindex just in case
        for i, t in enumerate(turns):
            t.turn_index = i

        if session_id in self._clarifications and self._clarifications[session_id].resolved:
            self.clear_pending_clarification(session_id)

        return turn

    def get_recent_context(self, session_id: str, limit: int = 3) -> str:
        """Return recent N turns formatted up to char limit."""
        if session_id not in self._sessions:
            return ""
        turns = self.get_session(session_id)
        recent = turns[-limit:] if limit > 0 else []

        # Format and truncate by char limit
        text = self._format_turns(recent)
        if len(text) > self.max_characters:
            text = text[-self.max_characters:]
        return text

    def get_relevant_context(self, session_id: str, query: str) -> str:
        """Select only overlapping history, or the latest turn for references."""
        if session_id not in self._sessions:
            return ""
        turns = self.get_session(session_id)
        if not turns:
            return ""

        # Use conversational entity extraction for better lowercase concept detection
        query_entities = set(e.lower() for e in self.qu.extract_conversational_entities(query))
        query_lower = query.lower()
        needs_recent_reference = bool(re.search(
            r"\b(?:it|its|this|that|they|them|the first one|the second one|the third one|"
            r"the previous method|the above process|the approach you mentioned|"
            r"next step|previous step|after that|following that)\b|^what about\b|^what's next\b",
            query_lower,
        ))

        scored_turns = []
        for i, turn in enumerate(turns):
            score = 0
            # Topic overlap - use active_topics if available
            turn_topics = set(
                topic.lower()
                for topic in turn.active_topics + turn.referenced_entities
            )
            if turn_topics:
                overlap = query_entities.intersection(turn_topics)
                score += len(overlap) * 2
            else:
                # Fallback to extracting from query and answer
                turn_text = f"{turn.query} {turn.answer}"
                turn_entities = set(e.lower() for e in self.qu.extract_conversational_entities(turn_text))
                overlap = query_entities.intersection(turn_entities)
                score += len(overlap) * 2

            # Referenced document overlap
            for doc in turn.referenced_documents:
                if doc.lower() in query.lower():
                    score += 3

            scored_turns.append((score, i, turn))

        # An explicitly named topic wins over recency. If a reference has no
        # entity/document match, include only its immediate antecedent.
        if needs_recent_reference and not any(score > 0 for score, _, _ in scored_turns):
            scored_turns = [
                (score + 10 if idx == len(turns) - 1 else score, idx, turn)
                for score, idx, turn in scored_turns
            ]

        # Sort by relevance first, then choose the newest turn for ties.
        scored_turns.sort(key=lambda x: (x[0], x[1]), reverse=True)

        selected_turns = []
        current_chars = 0

        for score, idx, turn in scored_turns:
            if score == 0:
                continue

            turn_str = self._format_turns([turn])
            if current_chars + len(turn_str) > self.max_characters:
                continue
            selected_turns.append((idx, turn))
            current_chars += len(turn_str)

        # Re-sort selected by chronological order
        selected_turns.sort(key=lambda x: x[0])
        return self._format_turns([t for _, t in selected_turns])

    def _format_turns(self, turns: List[ConversationTurn]) -> str:
        parts = []
        for t in turns:
            parts.append(f"User: {t.query}\nAssistant: {t.answer}")
        return "\n\n".join(parts)

    def set_pending_clarification(self, session_id: str, state: ClarificationState) -> None:
        self.create_session(session_id)
        count = self._clarification_counts[session_id]
        if count >= self.max_clarifications:
            raise ValueError("Maximum clarification loops reached for this turn.")

        state.created_at = time.time()
        self._clarifications[session_id] = state
        self._clarification_counts[session_id] = count + 1

    def get_pending_clarification(self, session_id: str) -> Optional[ClarificationState]:
        return self._clarifications.get(session_id)

    def clear_pending_clarification(self, session_id: str) -> None:
        self._clarifications.pop(session_id, None)
        if session_id in self._clarification_counts:
            self._clarification_counts[session_id] = 0

    def resolve_clarification(self, session_id: str, clarification_id: str, answer: str) -> str:
        state = self.get_pending_clarification(session_id)
        if not state:
            raise ValueError("No pending clarification for this session.")

        if state.resolved:
            raise ValueError("Clarification already resolved.")

        if state.clarification_id != clarification_id:
            raise ValueError("Invalid clarification_id.")

        answer_clean = answer.strip()
        if not answer_clean:
            raise ValueError("Clarification answer cannot be empty.")

        refined = f"{state.original_query} {answer_clean}".strip()
        explicit_meaning = re.search(r"\b(?:i mean|meaning)\s+(.+)$", answer_clean, re.IGNORECASE)
        if explicit_meaning:
            clarified_entity = explicit_meaning.group(1).strip().rstrip(".!?")
            refined = re.sub(
                r"\b(the agent|the process|the system|the workflow|the requirements)\b",
                clarified_entity,
                state.original_query,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            if refined == state.original_query.strip():
                refined = f"{state.original_query} {clarified_entity}".strip()

        state.refined_query = refined
        state.clarification_answer = answer_clean
        state.resolved = True
        for turn in reversed(self._sessions.get(session_id, [])):
            if (
                turn.query == state.original_query
                and turn.clarification_question == state.question
            ):
                turn.clarification_answer = answer_clean
                turn.refined_query = refined
                break
        return refined
