"""QueryUnderstandingAgent — deterministic M2 query classification."""

from __future__ import annotations

import re
from typing import List, Optional

from agents.models import QueryUnderstandingResult, QueryType

INTENTS = ("factual", "procedural", "comparative", "ambiguous")

_PROCEDURAL_PATTERNS = (
    r"\bhow (?:do|to|should|can)\b",
    r"\bwhat are the steps\b",
    r"\bprocedure\b",
    r"\bprotocol\b",
    r"\bworkflow\b",
    r"\bprocess\b",
    r"\bfollow\b",
    r"\bfive rights\b",
    r"\bmedication administration\b",
)

_COMPARATIVE_PATTERNS = (
    r"\bdiffer\b",
    r"\bdifference\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcontrast\b",
    r"\bsimilarities\b",
    r"\bsimilarities and differences\b",
)

_SOFTWARE_KEYWORDS = (
    "git", "sprint", "agile", "microservice", "programming", "code",
    "software", "branch", "pull request", "class", "logging", "rust", "go",
    "rest", "soap", "api", "http",
)

_HOSPITAL_KEYWORDS = (
    "patient", "nursing", "medication", "hospital", "admission",
    "code blue", "code red", "hand hygiene", "acetaminophen", "ibuprofen",
    "emergency", "dose",
)

_VAGUE_PRONOUNS = re.compile(r"\b(it|this|that|they|them)\b", re.I)

# Entity extraction: capitalised tokens, quoted strings, domain keywords
_QUOTED = re.compile(r'"([^"]+)"|\u2018([^\u2019]+)\u2019')
_CAPITALISED = re.compile(r'\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*)\b')
_DOMAIN_TERMS = frozenset(
    list(_SOFTWARE_KEYWORDS) + list(_HOSPITAL_KEYWORDS)
    + ["REST", "SOAP", "ICU", "API", "HTTP", "paracetamol", "ibuprofen"]
)


class QueryUnderstandingAgent:
    """Rule-based query understanding (no LLM)."""

    def normalize(self, query: str) -> str:
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def has_comparative_cue(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(pattern, q) for pattern in _COMPARATIVE_PATTERNS)

    def classify_intent(self, query: str) -> QueryType:
        q = query.lower()
        if self.is_ambiguous(query):
            return "ambiguous"
        if self.has_comparative_cue(query):
            return "comparative"
        if any(re.search(p, q) for p in _PROCEDURAL_PATTERNS):
            return "procedural"
        return "factual"

    def _classification_details(
        self,
        query_type: QueryType,
    ) -> tuple[float, str]:
        if query_type == "ambiguous":
            return (
                0.9,
                "The query is too short, vague, or contains an unresolved referent.",
            )
        if query_type == "comparative":
            return 0.95, "A comparison cue identifies a comparative request."
        if query_type == "procedural":
            return 0.95, "A procedure cue identifies a request for steps or process."
        return 0.8, "No procedural or comparative cue was found; the query requests information."

    def detect_domain(self, query: str) -> Optional[str]:
        q = query.lower()
        software = sum(1 for kw in _SOFTWARE_KEYWORDS if kw in q)
        hospital = sum(1 for kw in _HOSPITAL_KEYWORDS if kw in q)
        if software > hospital and software > 0:
            return "Software Engineering"
        if hospital > software and hospital > 0:
            return "Hospital Administration"
        return None

    def is_ambiguous(self, query: str) -> bool:
        words = query.split()
        if not words:
            return True
        if len(words) <= 2:
            return True
        if len(words) <= 4 and _VAGUE_PRONOUNS.search(query):
            return True
        if query.endswith("?") and len(words) <= 3:
            return not self.extract_conversational_entities(query)
        return False

    def extract_entities(self, query: str) -> List[str]:
        """Return a deduplicated list of meaningful entities for M1 callers."""
        entities: List[str] = []
        for match in _QUOTED.finditer(query):
            entities.append((match.group(1) or match.group(2)).strip())
        query_lower = re.sub(r"[.!?]+\s*$", "", query.strip().lower())
        for term in sorted(_DOMAIN_TERMS):
            if term.lower() in query_lower:
                entities.append(term)
        for match in _CAPITALISED.finditer(query):
            token = match.group(1)
            if token not in {"The", "What", "How", "Who", "When", "Where", "Why"}:
                entities.append(token)

        seen = set()
        result: List[str] = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                result.append(entity)
        return result

    def extract_comparison_targets(self, query: str) -> List[str]:
        """Extract comparison targets from comparative queries.

        Handles natural comparative patterns:
        - "Compare A and B"
        - "Compare A with B"
        - "Compare A versus B"
        - "What is the difference between A and B?"
        - "How does A differ from B?"
        - "A vs B"
        - "Compare the advantages of A and B"

        Returns empty list if no valid targets found or if targets are pronouns-only.
        Works regardless of capitalization and handles multi-word targets.
        """
        query_lower = query.lower()
        targets: List[str] = []

        # Pattern 1: "Compare A and B", "Compare A with B", "Compare A versus B"
        compare_patterns = [
            r"compare\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"compare\s+(.+?)\s+with\s+(.+?)(?:\?|$)",
            r"compare\s+(.+?)\s+versus\s+(.+?)(?:\?|$)",
            r"compare\s+(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        ]

        for pattern in compare_patterns:
            match = re.search(pattern, query_lower)
            if match:
                target1 = match.group(1).strip()
                target2 = match.group(2).strip()
                # Filter out leading phrases like "the advantages of", "a disadvantage of", etc.
                target1 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target1).strip()
                target2 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target2).strip()
                # Filter out articles
                target1 = re.sub(r'^(?:the|a|an)\s+', '', target1).strip()
                target2 = re.sub(r'^(?:the|a|an)\s+', '', target2).strip()
                # Filter out trailing words
                target1 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target1).strip()
                target2 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target2).strip()
                if target1 and target2:
                    targets.extend([target1, target2])
                break

        # Pattern 2: "What is the difference between A and B?"
        if not targets:
            diff_pattern = r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)"
            match = re.search(diff_pattern, query_lower)
            if match:
                target1 = match.group(1).strip()
                target2 = match.group(2).strip()
                # Filter out leading phrases like "the advantages of", etc.
                target1 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target1).strip()
                target2 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target2).strip()
                # Filter out articles
                target1 = re.sub(r'^(?:the|a|an)\s+', '', target1).strip()
                target2 = re.sub(r'^(?:the|a|an)\s+', '', target2).strip()
                # Filter out trailing words
                target1 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target1).strip()
                target2 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target2).strip()
                if target1 and target2:
                    targets.extend([target1, target2])

        # Pattern 3: "How does A differ from B?" or "How do A differ from B?"
        if not targets:
            differ_patterns = [
                r"how\s+does\s+(.+?)\s+differ\s+from\s+(.+?)(?:\?|$)",
                r"how\s+do\s+(.+?)\s+differ\s+from\s+(.+?)(?:\?|$)",
            ]
            for differ_pattern in differ_patterns:
                match = re.search(differ_pattern, query_lower)
                if match:
                    target1 = match.group(1).strip()
                    target2 = match.group(2).strip()
                    # Filter out leading phrases like "the advantages of", etc.
                    target1 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target1).strip()
                    target2 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target2).strip()
                    # Filter out articles
                    target1 = re.sub(r'^(?:the|a|an)\s+', '', target1).strip()
                    target2 = re.sub(r'^(?:the|a|an)\s+', '', target2).strip()
                    # Filter out trailing words
                    target1 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target1).strip()
                    target2 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target2).strip()
                    if target1 and target2:
                        targets.extend([target1, target2])
                    break

        # Pattern 4: "A vs B" or "A versus B" (when compare is implied)
        if not targets:
            vs_patterns = [
                r"^(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
                r"^(.+?)\s+versus\s+(.+?)(?:\?|$)",
            ]
            for pattern in vs_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    target1 = match.group(1).strip()
                    target2 = match.group(2).strip()
                    # Filter out leading phrases like "the advantages of", etc.
                    target1 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target1).strip()
                    target2 = re.sub(r'^(?:the\s+)?(?:a\s+)?(?:an\s+)?(?:advantages?|disadvantages?|benefits?)\s+of\s+', '', target2).strip()
                    # Filter out articles
                    target1 = re.sub(r'^(?:the|a|an)\s+', '', target1).strip()
                    target2 = re.sub(r'^(?:the|a|an)\s+', '', target2).strip()
                    # Filter out trailing words
                    target1 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target1).strip()
                    target2 = re.sub(r'\s+(?:in|for|of|the|a|an)\s*$', '', target2).strip()
                    if target1 and target2:
                        targets.extend([target1, target2])
                    break

        # Filter out pronoun-only targets
        pronouns = {
            "it", "this", "that", "they", "them", "he", "she", "one", "ones",
            "two", "both",
        }
        filtered_targets = []
        for target in targets:
            # Check if target is just a pronoun or contains only pronouns
            target = target.strip(" \t.,!?;:")
            words = target.split()
            if not all(word.lower() in pronouns for word in words):
                if target:
                    filtered_targets.append(target)

        return filtered_targets

    def extract_conversational_entities(self, query: str) -> List[str]:
        """Extract entities suitable for conversation context resolution.

        Unlike extract_entities, this method is more permissive to capture
        lowercase document concepts that appear in natural conversation.
        Returns meaningful entities that could serve as conversation topics.
        """
        entities: List[str] = []
        query_lower = query.lower()

        # 1. Quoted strings (highest priority)
        for match in _QUOTED.finditer(query):
            entities.append((match.group(1) or match.group(2)).strip())

        # 2. Domain keywords (from predefined list)
        for term in sorted(_DOMAIN_TERMS):
            if term.lower() in query_lower:
                entities.append(term)

        # 3. Capitalized tokens (proper nouns, acronyms)
        _NOISE_CAPITALIZED = {"The", "What", "How", "Who", "When", "Where", "Why", "Explain", "Compare", "Tell", "Summarize", "User", "Assistant", "Semantic", "Approach", "Mentioned"}
        for match in _CAPITALISED.finditer(query):
            token = match.group(1)
            if token not in _NOISE_CAPITALIZED:
                entities.append(token)

        def _keep_entity_phrase(phrase: str) -> bool:
            text = phrase.strip(" \t.,!?;:")
            if not text or len(text.split()) > 4:
                return False
            if re.search(r"\b(?:is|are|was|were|be|been|being|a|an|the|of|to|for|with|about|what|how|why|who|when|where|explain|tell|me|this|that|it|its|they|them|you|we|i)\b", text, flags=re.I):
                return False
            return True

        # 4. Multi-word technical concepts (lowercase)
        # Look for patterns like "semantic versioning", "jwt authentication", etc.
        # These are 2-4 word phrases that could be technical concepts
        technical_patterns = [
            r'\b(?:semantic|jwt|oauth|rest|soap|api|http|https|ssl|tls|git|docker|kubernetes|microservice|monolithic|agile|scrum|kanban|continuous|integration|delivery|deployment|authentication|authorization|encryption|hashing|database|sql|nosql|cache|queue|message|broker|server|client|frontend|backend|fullstack|devops|ci|cd|release|train|branch|merge|rebase|commit|push|pull|request|review|testing|unit|integration|e2e|acceptance|performance|load|stress|security|vulnerability|penetration|audit|compliance|regulation|standard|protocol|format|json|xml|yaml|toml|ini|config|setting|parameter|argument|option|flag|switch|environment|variable|constant|function|method|class|object|interface|abstract|inheritance|polymorphism|encapsulation|abstraction|solid|dry|kiss|yagni|clean|code|smell|refactor|pattern|design|architecture|system|service|component|module|library|framework|package|dependency|version|control|repository|workflow|pipeline|stage|step|process|workflow|methodology|approach|strategy|technique|practice|principle|guideline|best|practice|anti|pattern|solution|problem|challenge|issue|bug|error|exception|failure|success|result|outcome|benefit|advantage|disadvantage|pro|con|trade|off|consideration|factor|aspect|feature|capability|limitation|constraint|requirement|specification|documentation|comment|description|explanation|example|sample|demo|tutorial|guide|reference|manual|handbook|standard|spec|protocol|format|schema|model|data|structure|algorithm|logic|flow|sequence|order|priority|urgency|importance|relevance|significance|impact|effect|influence|cause|consequence|result|outcome|output|input|parameter|argument|return|value|type|kind|sort|category|class|group|set|collection|list|array|map|dictionary|hash|table|record|row|column|field|attribute|property|characteristic|quality|quantity|amount|number|count|measure|metric|indicator|stat|figure|data|information|knowledge|wisdom|insight|understanding|comprehension|grasp|awareness|consciousness|perception|observation|detection|recognition|identification|classification|categorization|grouping|organization|arrangement|structure|pattern|relationship|connection|association|link|tie|bond|chain|network|graph|tree|hierarchy|level|layer|tier|stage|phase|step|process|workflow|pipeline|sequence|order|series|chain|stream|flow|current|trend|direction|movement|change|evolution|development|progress|advancement|improvement|enhancement|optimization|refinement|adjustment|modification|alteration|adaptation|transformation|conversion|translation|interpretation|explanation|clarification|elaboration|specification|definition|description|characterization|representation|presentation|display|visualization|rendering|drawing|diagram|chart|graph|plot|map|plan|design|layout|structure|architecture|framework|model|pattern|template|blueprint|schema|format|standard|convention|rule|principle|guideline|policy|procedure|process|method|technique|approach|strategy|tactic|plan|schedule|timeline|deadline|milestone|goal|objective|target|aim|purpose|intention|motivation|reason|cause|factor|driver|enabler|barrier|obstacle|challenge|problem|issue|risk|threat|opportunity|benefit|value|worth|merit|importance|significance|relevance|priority|urgency|impact|effect|influence|contribution|role|responsibility|accountability|authority|ownership|control|management|governance|oversight|supervision|coordination|collaboration|cooperation|communication|interaction|exchange|transfer|transmission|delivery|distribution|allocation|assignment|delegation|empowerment|enablement|facilitation|support|assistance|help|aid|service|function|operation|activity|action|task|job|work|effort|energy|resource|asset|capability|capacity|ability|skill|competence|expertise|knowledge|experience|insight|wisdom|intelligence|understanding|awareness|consciousness|mindset|attitude|behavior|practice|habit|routine|culture|environment|context|situation|condition|circumstance|scenario|case|example|instance|occurrence|event|incident|accident|failure|error|mistake|fault|defect|bug|issue|problem|challenge|difficulty|obstacle|barrier|hurdle|impediment|constraint|limitation|restriction|requirement|specification|criterion|standard|expectation|demand|need|want|desire|preference|choice|option|alternative|possibility|opportunity|chance|risk|threat|danger|hazard|exposure|vulnerability|weakness|gap|shortcoming|deficiency|lack|absence|missing|lost|forgotten|ignored|neglected|overlooked|missed|skipped|omitted|excluded|rejected|denied|refused|declined|accepted|approved|authorized|permitted|allowed|enabled|supported|endorsed|recommended|suggested|proposed|offered|provided|supplied|delivered|given|granted|awarded|assigned|allocated|distributed|shared|divided|split|separated|partitioned|organized|arranged|structured|ordered|sorted|ranked|rated|scored|evaluated|assessed|measured|quantified|qualified|classified|categorized|grouped|clustered|organized|structured|arranged|ordered|sorted|ranked|rated|scored|evaluated|assessed|measured|quantified|qualified|classified|categorized|grouped|clustered)\\s+(?:versioning|authentication|authorization|encryption|hashing|signature|validation|verification|certification|accreditation|registration|subscription|membership|participation|involvement|engagement|interaction|communication|collaboration|cooperation|coordination|integration|connection|association|relationship|link|tie|bond|chain|network|graph|tree|hierarchy|structure|system|framework|platform|environment|ecosystem|landscape|market|industry|sector|domain|field|area|discipline|subject|topic|theme|concept|idea|notion|thought|belief|opinion|view|perspective|standpoint|position|stance|attitude|approach|method|technique|strategy|tactic|plan|design|architecture|blueprint|specification|requirement|constraint|limitation|assumption|hypothesis|theory|principle|law|rule|regulation|standard|guideline|policy|procedure|process|workflow|pipeline|sequence|series|chain|flow|stream|current|trend|movement|direction|change|evolution|development|progress|advancement|improvement|enhancement|optimization|refinement|adjustment|modification|alteration|adaptation|transformation|conversion|translation|interpretation|explanation|clarification|elaboration|specification|definition|description|characterization|representation|presentation|display|visualization|rendering|drawing|diagram|chart|graph|plot|map|plan|layout|structure|framework|model|pattern|template|blueprint|schema|format|standard|convention)\b',
        ]

        for pattern in technical_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                phrase = match.group(0).strip()
                if _keep_entity_phrase(phrase) and phrase not in entities:
                    entities.append(phrase)

        # 5. Common technical compound terms (2-3 words)
        common_compounds = [
            "semantic versioning", "jwt", "jwt authentication", "oauth authorization", "rest api", "soap api",
            "http request", "https request", "ssl certificate", "tls handshake", "git branch", "git commit",
            "docker container", "kubernetes pod", "microservice architecture", "monolithic architecture",
            "agile methodology", "scrum framework", "kanban board", "continuous integration", "continuous delivery",
            "ci/cd pipeline", "deployment pipeline", "release train", "release trains", "feature branch", "pull request",
            "code review", "unit test", "integration test", "e2e test", "load test", "stress test",
            "security audit", "penetration test", "compliance check", "best practice", "design pattern",
            "anti pattern", "clean code", "solid principles", "dry principle", "kiss principle",
            "yagni principle", "code smell", "technical debt", "refactoring", "architecture decision",
            "system design", "database schema", "api specification", "data model", "algorithm design",
            "flow chart", "sequence diagram", "use case", "user story", "acceptance criteria",
            "definition of done", "definition of ready", "sprint planning", "daily standup",
            "sprint review", "sprint retrospective", "backlog refinement", "story points",
            "velocity tracking", "burndown chart", "cumulative flow", "lead time", "cycle time",
            "throughput", "bottleneck", "constraint", "dependency", "critical path",
            "risk assessment", "impact analysis", "root cause analysis", "post mortem",
            "lessons learned", "action item", "follow up", "action plan", "mitigation strategy",
            "contingency plan", "rollback plan", "disaster recovery", "business continuity",
            "service level agreement", "service level objective", "operational level agreement",
            "key performance indicator", "service level indicator", "objectives and key results",
            "balanced scorecard", "swot analysis", "pest analysis", "porter five forces",
            "value proposition", "business model canvas", "lean canvas", "customer journey map",
            "user experience", "user interface", "customer experience", "brand experience",
            "touchpoint", "channel", "platform", "device", "operating system", "browser",
            "mobile app", "web app", "desktop app", "native app", "hybrid app", "progressive web app",
            "single page application", "multi page application", "server side rendering", "client side rendering",
            "static site generator", "headless cms", "content management system", "digital asset management",
            "customer relationship management", "enterprise resource planning", "supply chain management",
            "inventory management", "order management", "payment processing", "fraud detection",
            "identity verification", "access control", "permission management", "role based access control",
            "attribute based access control", "multi factor authentication", "single sign on",
            "federated identity", "identity provider", "service provider", "relying party",
            "assertion based access control", "policy based access control", "mandatory access control",
            "discretionary access control", "role based access control", "rule based access control"
        ]

        for compound in common_compounds:
            if compound in query_lower and _keep_entity_phrase(compound):
                if compound not in entities:
                    entities.append(compound)

        topic_patterns = (
            r"^(?:what\s+(?:is|are)|explain|describe|define|tell\s+me\s+about)\s+(.+?)[?.!]*$",
            r"^what\s+about\s+(.+?)[?.!]*$",
        )
        generic_topics = {
            "agent", "process", "processes", "system", "workflow", "requirements",
            "topic", "thing", "things", "step", "steps", "method", "approach",
        }
        for line in query.splitlines():
            topic_query = re.sub(r"^(?:User|Assistant):\s*", "", line.strip(), flags=re.I)
            for pattern in topic_patterns:
                match = re.match(pattern, topic_query, flags=re.I)
                if not match:
                    continue
                topic = match.group(1).strip().strip(" \t.,!?;:")
                topic = re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.I)
                if (
                    topic
                    and topic.lower() not in generic_topics
                    and _keep_entity_phrase(topic)
                    and not re.match(
                        r"^(?:it|its|they|them|this|that|(?:the\s+)?(?:first|second|third)\s+one)\b",
                        topic,
                        flags=re.I,
                    )
                ):
                    entities.append(topic)
                break

        for match in re.finditer(
            r"\b(?:are|include|includes|such as)\s+(.+?)(?:[.!?]|$)",
            query,
            flags=re.I,
        ):
            items = re.split(
                r",\s*(?:and|or)\s+|,\s*|\s+(?:and|or)\s+",
                match.group(1),
            )
            if len(items) < 2:
                continue
            for item in items:
                item = re.sub(r"^(?:the|a|an)\s+", "", item.strip(), flags=re.I)
                item = item.strip(" \t.,!?;:")
                if item and len(item.split()) <= 5 and _keep_entity_phrase(item):
                    entities.append(item)

        # Deduplicate while preserving order
        seen = set()
        result: List[str] = []
        for entity in entities:
            entity_lower = entity.lower()
            # Remove leading articles for comparison
            entity_normalized = re.sub(r'^(the|a|an)\s+', '', entity_lower).strip()
            if entity_normalized not in seen:
                seen.add(entity_normalized)
                result.append(entity)

        return result

    def analyze(self, query: str) -> QueryUnderstandingResult:
        normalized = self.normalize(query)
        query_type = self.classify_intent(normalized)
        confidence, reason = self._classification_details(query_type)
        return QueryUnderstandingResult(
            query=query,
            normalized_query=normalized,
            query_type=query_type,
            classification_confidence=confidence,
            routing="CLARIFICATION" if query_type == "ambiguous" else "RETRIEVAL",
            domain=self.detect_domain(normalized),
            reason=reason,
        )
