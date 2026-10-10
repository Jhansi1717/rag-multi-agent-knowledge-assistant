"""Contextual Query Resolution service."""

import re
from typing import List, Optional

from agents.models import ClarificationState, ContextResolutionResult
from agents.query_understanding import QueryUnderstandingAgent


# Words that are capitalized in natural speech but carry no entity meaning
_NOISE_WORDS = frozenset({
    "tell", "me", "about", "now", "hello", "hi", "there", "yes", "no", "okay",
    "ok", "sure", "thanks", "please", "sorry", "great", "good", "just",
    "it", "is", "are", "the", "and", "but", "for", "with", "that", "this",
    "what", "how", "why", "who", "when", "where", "explain", "summarize",
    "compare", "both", "tools", "also", "can", "could", "would", "should",
    "use", "used", "using", "want", "need", "like", "know", "think",
    "user", "assistant", "semantic", "approach", "mentioned",
})


class ContextResolutionAgent:
    """Resolves conversational references using recent context deterministically."""

    def __init__(self, qu_agent: Optional[QueryUnderstandingAgent] = None):
        self.qu = qu_agent or QueryUnderstandingAgent()

    @staticmethod
    def _is_topic_phrase(candidate: str) -> bool:
        text = candidate.strip(" \t.,!?;:")
        if not text or len(text.split()) > 4:
            return False
        if re.search(r"\b(?:is|are|was|were|be|been|being|a|an|the|of|to|for|with|about|what|how|why|who|when|where|explain|tell|me|this|that|it|its|they|them|you|we|i)\b", text, flags=re.I):
            return False
        return True

    def _clean_entities(self, context: str) -> List[str]:
        """Extract entities from context, filtering out noise words."""
        clean = context.replace("User:", "").replace("Assistant:", "")
        raw = []
        for line in clean.splitlines():
            line = line.strip()
            if not line:
                continue
            raw.extend(
                e for e in self.qu.extract_conversational_entities(line)
                if self._is_topic_phrase(e)
            )
        seen = set()
        result: List[str] = []
        for entity in raw:
            key = entity.lower().strip(" \t.,!?;:")
            if key and key not in seen:
                seen.add(key)
                result.append(entity.strip(" \t.,!?;:"))
        return [e for e in result if e.lower() not in _NOISE_WORDS and len(e) > 2]

    def _extract_topics_from_context(self, context: str) -> List[str]:
        """Extract topics from context string using conversational entity extraction."""
        clean = context.replace("User:", "").replace("Assistant:", "")
        entities = []
        for line in clean.splitlines():
            line = line.strip()
            if not line:
                continue
            entities.extend(
                e for e in self.qu.extract_conversational_entities(line)
                if self._is_topic_phrase(e)
            )
        deduped: List[str] = []
        seen = set()
        for entity in entities:
            key = entity.lower().strip(" \t.,!?;:")
            if key and key not in seen:
                seen.add(key)
                deduped.append(entity.strip(" \t.,!?;:"))
        return [e for e in deduped if e.lower() not in _NOISE_WORDS and len(e) > 2]

    @staticmethod
    def _single_unambiguous_topic(topics: List[str]) -> Optional[str]:
        unique_topics = list(dict.fromkeys(
            topic.strip()
            for topic in topics
            if topic.strip() and ContextResolutionAgent._is_topic_phrase(topic)
        ))
        if not unique_topics:
            return None
        topic_tokens = {
            topic: {
                token[:-3] + "y" if token.endswith("ies")
                else token[:-2] if token.endswith(("ches", "shes", "sses", "xes", "zes"))
                else token[:-1] if token.endswith("s") and not token.endswith(("ss", "us", "is"))
                else token
                for token in re.findall(r"[a-z0-9]+", topic.lower())
            }
            for topic in unique_topics
        }
        longest = max(unique_topics, key=lambda topic: (len(topic_tokens[topic]), len(topic)))
        if all(tokens <= topic_tokens[longest] for tokens in topic_tokens.values()):
            return longest
        return None

    @staticmethod
    def _capitalize_first(text: str) -> str:
        return text[:1].upper() + text[1:]

    def resolve(
        self,
        query: str,
        context: str,
        clarification_state: Optional[ClarificationState] = None,
        active_topics: Optional[List[str]] = None,
        referenced_entities: Optional[List[str]] = None,
    ) -> ContextResolutionResult:
        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # Pending clarification takes priority — don't rewrite
        if clarification_state and not clarification_state.resolved:
            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="Pending clarification takes precedence.",
            )

        if not context:
            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="No context available.",
            )

        pronouns = [r"\bits\b", r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthey\b", r"\bthem\b"]
        has_pronoun = any(re.search(p, query_lower) for p in pronouns)

        refs = [r"the first one", r"the second one", r"the third one", r"the previous method", r"the above process", r"the approach you mentioned"]
        has_ref = False
        ref_match = ""
        for r in refs:
            if re.search(r, query_lower):
                has_ref = True
                ref_match = r
                break

        is_continuation = query_lower.startswith("what about ")

        # Standalone query — nothing to resolve
        if not (has_pronoun or has_ref or is_continuation):
            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="Query is standalone and clear.",
            )

        context_entities = self._clean_entities(context)
        current_topics = [
            topic for topic in (active_topics or [])
            if topic.strip()
            and topic.strip().lower() not in _NOISE_WORDS
            and self._is_topic_phrase(topic)
        ]
        reference_topics = referenced_entities or current_topics or context_entities
        query_topics = self._clean_entities(query_stripped)
        known_topics = {
            topic.strip().lower()
            for topic in [*current_topics, *reference_topics, *context_entities]
            if topic.strip()
        }
        if any(topic.strip().lower() in known_topics for topic in query_topics):
            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="Query explicitly names a known topic.",
            )

        # Check for explicit references BEFORE continuation (e.g., "What about the second one?")
        if has_ref:
            if ref_match == r"the first one":
                if len(reference_topics) >= 1:
                    resolved = re.sub(ref_match, reference_topics[0], query_stripped, flags=re.I)
                    return ContextResolutionResult(
                        original_query=query_stripped,
                        resolved_query=self._capitalize_first(resolved),
                        context_used=True,
                        resolution_reason="Resolved numbered reference.",
                        referenced_entities=[reference_topics[0]],
                    )
            elif ref_match == r"the second one":
                if len(reference_topics) >= 2:
                    resolved = re.sub(ref_match, reference_topics[1], query_stripped, flags=re.I)
                    return ContextResolutionResult(
                        original_query=query_stripped,
                        resolved_query=self._capitalize_first(resolved),
                        context_used=True,
                        resolution_reason="Resolved numbered reference.",
                        referenced_entities=[reference_topics[1]],
                    )
            elif ref_match == r"the third one":
                if len(reference_topics) >= 3:
                    resolved = re.sub(ref_match, reference_topics[2], query_stripped, flags=re.I)
                    return ContextResolutionResult(
                        original_query=query_stripped,
                        resolved_query=self._capitalize_first(resolved),
                        context_used=True,
                        resolution_reason="Resolved numbered reference.",
                        referenced_entities=[reference_topics[2]],
                    )
            elif ref_match in [r"the previous method", r"the above process", r"the approach you mentioned"]:
                topics = current_topics or context_entities
                topic = self._single_unambiguous_topic(topics)
                if topic:
                    resolved = re.sub(ref_match, topic, query_stripped, flags=re.I)
                    return ContextResolutionResult(
                        original_query=query_stripped,
                        resolved_query=self._capitalize_first(resolved),
                        context_used=True,
                        resolution_reason="Resolved previous topic reference.",
                        referenced_entities=[topic],
                    )
            # If explicit reference fails, don't fall through to continuation
            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="Insufficient context to resolve explicit reference.",
            )

        # --- Topic continuation: "What about X?" ---
        if is_continuation:
            topics = current_topics or context_entities
            topic = self._single_unambiguous_topic(topics)
            if topic:
                return ContextResolutionResult(
                    original_query=query_stripped,
                    resolved_query=f"{query_stripped} in the context of {topic}",
                    context_used=True,
                    resolution_reason="Appended previous topic to continuation query.",
                    referenced_entities=[topic],
                )
            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="Insufficient context to resolve continuation.",
            )

        # --- Pronoun reference: "it", "this", "that", "they", "them" ---
        if has_pronoun:
            # Prioritize the most recent turn for single-topic resolution
            # Parse context to get the last turn's topics
            context_lines = context.strip().split('\n')
            last_user_line = None
            last_assistant_line = None
            for line in reversed(context_lines):
                if line.startswith('User:') and last_user_line is None:
                    last_user_line = line.replace('User:', '').strip()
                elif line.startswith('Assistant:') and last_assistant_line is None:
                    last_assistant_line = line.replace('Assistant:', '').strip()

            if last_user_line:
                last_turn_entities = current_topics or self._extract_topics_from_context(
                    last_user_line
                )
                topic = self._single_unambiguous_topic(last_turn_entities)
                if not topic and not current_topics and last_assistant_line:
                    answer_entities = self._extract_topics_from_context(
                        last_assistant_line
                    )
                    topic = self._single_unambiguous_topic(answer_entities)
                if topic:
                    for p in pronouns:
                        if re.search(p, query_lower):
                            topic_lower = topic.lower()
                            replacement = (
                                topic_lower + ("'" if topic_lower.endswith("s") else "'s")
                                if p == r"\bits\b"
                                else topic_lower
                            )
                            resolved = re.sub(p, replacement, query_stripped, count=1, flags=re.I)
                            return ContextResolutionResult(
                                original_query=query_stripped,
                                resolved_query=self._capitalize_first(resolved),
                                context_used=True,
                                resolution_reason="Resolved pronoun reference to most recent topic.",
                                referenced_entities=[topic],
                            )

            # Fallback: if the most recent turn doesn't have a single topic,
            # check if there is exactly one clear entity in the entire context
            if not current_topics and len(context_entities) == 1:
                topic = context_entities[0]
                for p in pronouns:
                    if re.search(p, query_lower):
                        topic_lower = topic.lower()
                        replacement = (
                            topic_lower + ("'" if topic_lower.endswith("s") else "'s")
                            if p == r"\bits\b"
                            else topic_lower
                        )
                        resolved = re.sub(p, replacement, query_stripped, count=1, flags=re.I)
                        return ContextResolutionResult(
                            original_query=query_stripped,
                            resolved_query=self._capitalize_first(resolved),
                            context_used=True,
                            resolution_reason="Resolved pronoun reference.",
                            referenced_entities=[topic],
                        )

            return ContextResolutionResult(
                original_query=query_stripped,
                resolved_query=query_stripped,
                context_used=False,
                resolution_reason="Insufficient context to reliably resolve pronoun.",
            )

        return ContextResolutionResult(
            original_query=query_stripped,
            resolved_query=query_stripped,
            context_used=False,
            resolution_reason="Failed to resolve.",
        )
