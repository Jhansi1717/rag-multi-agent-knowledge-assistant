"""ClarificationAgent — handle vague or low-confidence queries deterministically."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from agents.models import ClarificationResult
from agents.query_understanding import QueryUnderstandingAgent


class ClarificationAgent:
    """Evaluate ambiguity and generate targeted clarification questions."""

    def __init__(self, query_understanding_agent: Optional[QueryUnderstandingAgent] = None):
        self.qu = query_understanding_agent or QueryUnderstandingAgent()

    def analyze(self, query: str) -> ClarificationResult:
        original = query
        query_stripped = query.strip()

        # 1. Empty/Whitespace
        if not query_stripped:
            return ClarificationResult(
                original_query=original,
                clarification_needed=True,
                ambiguity_type="incomplete",
                reason="Query is empty or whitespace only.",
                clarification_question="What would you like to know?",
                clarification_id=str(uuid.uuid4())
            )

        query_lower = query_stripped.lower()

        # 2. Multi-part request
        multi_part_patterns = [
            r"\band\s+.*?compare\b",
            r"\band\s+.*?explain\b",
            r"\balso\s+.*?how\b",
            r"\bfirst\s+.*?then\b"
        ]
        if any(re.search(p, query_lower) for p in multi_part_patterns):
            return ClarificationResult(
                original_query=original,
                clarification_needed=True,
                ambiguity_type="multi_part",
                reason="Query appears to contain multiple distinct requests.",
                clarification_question="You've asked about multiple topics. Which one would you like to address first?",
                clarification_id=str(uuid.uuid4())
            )

        # 3. Context-dependent request
        context_patterns = [
            r"\bnext step\b",
            r"\bprevious step\b",
            r"\bafter that\b",
            r"\bfollowing that\b",
            r"\bwhat's next\b"
        ]
        if any(re.search(p, query_lower) for p in context_patterns):
            return ClarificationResult(
                original_query=original,
                clarification_needed=True,
                ambiguity_type="context_dependent",
                reason="Query references previous or relative steps.",
                clarification_question="Which specific process or previous step are you referring to?",
                clarification_id=str(uuid.uuid4())
            )

        is_comparative = self.qu.has_comparative_cue(query_stripped)
        comparison_targets = (
            self.qu.extract_comparison_targets(query_stripped)
            if is_comparative else []
        )
        if len(comparison_targets) >= 2:
            return ClarificationResult(
                original_query=original,
                clarification_needed=False
            )

        entities = self.qu.extract_entities(query_stripped)

        # 4. Unresolved reference
        # Context resolution handles pronouns (it, its, this, that, they, them)
        # But numbered references without entities should still be flagged
        numbered_ref_patterns = [
            r"\bthe first one\b",
            r"\bthe second one\b",
            r"\bthe third one\b"
        ]
        # Use conversational entity extraction for better coverage
        conversational_entities = self.qu.extract_conversational_entities(query_stripped)
        unresolved_reference_patterns = [
            r"\bits\b", r"\bit\b", r"\bthey\b", r"\bthem\b", r"\bthis\b", r"\bthat\b",
            r"\bthe first one\b", r"\bthe second one\b", r"\bthe third one\b",
            r"\bthe previous method\b", r"\bthe above process\b", r"\bthe approach you mentioned\b",
        ]
        if len(conversational_entities) == 0:
            for p in unresolved_reference_patterns:
                match = re.search(p, query_lower)
                if match:
                    ref = match.group(0)
                    return ClarificationResult(
                        original_query=original,
                        clarification_needed=True,
                        ambiguity_type=(
                            "unresolved_reference"
                            if p not in {r"\bits\b", r"\bit\b", r"\bthey\b", r"\bthem\b", r"\bthis\b", r"\bthat\b"}
                            else "incomplete"
                        ),
                        reason="Query contains an unresolved reference.",
                        clarification_question=f"Could you clarify what '{ref}' refers to?",
                        clarification_id=str(uuid.uuid4())
                    )

        # 5. Ambiguous terminology
        ambiguous_terms = [
            r"\bthe process\b",
            r"\bthe requirements\b",
            r"\bthe system\b",
            r"\bthe workflow\b"
        ]
        for term_p in ambiguous_terms:
            match = re.search(term_p, query_lower)
            if match:
                term = match.group(0)
                # Ensure no domain keywords are present to qualify it
                domain = self.qu.detect_domain(query_stripped)
                if not domain:
                    term_singular = term.replace('the ', '')
                    term_plural = {
                        "process": "processes",
                        "requirements": "requirements",
                        "system": "systems",
                        "workflow": "workflows",
                    }.get(term_singular, term_singular)
                    return ClarificationResult(
                        original_query=original,
                        clarification_needed=True,
                        ambiguity_type="ambiguous_terminology",
                        reason="Query uses generic terminology without domain context.",
                        clarification_question=f"There are multiple {term_plural} documented. Which specific one do you mean?",
                        clarification_id=str(uuid.uuid4())
                    )

        # 6. Missing object / Incomplete request

        if is_comparative:
            return ClarificationResult(
                original_query=original,
                clarification_needed=True,
                ambiguity_type="incomplete",
                reason="Comparative query lacks specific comparison targets.",
                clarification_question="What specific items would you like me to compare?",
                clarification_id=str(uuid.uuid4())
            )

        # For non-comparative queries, check for missing objects with other verbs
        action_verbs = [
            r"\bexplain\b",
            r"\bsummarize\b",
            r"\bwhat are\b",
            r"\btell me about\b",
        ]

        # Capitalized verbs might be extracted as entities.
        # Filter them out before checking for missing objects.
        verb_keywords = {"explain", "summarize", "what", "are", "tell"}
        filtered_entities = [
            e for e in conversational_entities if e.lower() not in verb_keywords
        ]

        if len(filtered_entities) == 0:
            for verb_p in action_verbs:
                match = re.search(verb_p, query_lower)
                if match:
                    verb = match.group(0)
                    if verb == "what are":
                        verb = "explain"
                    elif verb == "tell me about":
                        verb = "explain"
                    return ClarificationResult(
                        original_query=original,
                        clarification_needed=True,
                        ambiguity_type="incomplete",
                        reason="Query contains an action but no target entities.",
                        clarification_question=f"What specific topic or items would you like me to {verb}?",
                        clarification_id=str(uuid.uuid4())
                    )

        # 7. Clear query
        return ClarificationResult(
            original_query=original,
            clarification_needed=False
        )
