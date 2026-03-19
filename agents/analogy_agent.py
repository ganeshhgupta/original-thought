"""
agents/analogy_agent.py — Analogy Agent.

The Analogy Agent finds structural parallels between the current proof
obligation and problems in other mathematical domains.

IMPORTANT: The agent queries the corpus by ALGEBRAIC STRUCTURE DESCRIPTION,
never by topic name or technique name that might directly encode knowledge
of the target result. This preserves the experimental validity.

The agent fires when:
  1. Called explicitly (every ANALOGY_ACTIVATION_ROUND consecutive lean-flat rounds)
  2. Called during redirect protocol (Layer 2)

What this agent models:
  Most major mathematical discoveries involved recognizing that two apparently
  different problems were instances of the same deeper structure:
    - Fourier analysis and quantum mechanics share the same L² operator theory
    - Graph coloring and constraint satisfaction are the same problem
    - The Jones polynomial and quantum field theory share algebraic structure
    - Integer partitions and Young tableaux are the same combinatorial object under different notation

The agent describes the CURRENT proof obligation's algebraic structure
in abstract terms, retrieves structurally similar results from the corpus,
and proposes whether the technique from the analogous domain might work here.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from tools.corpus_retriever import CorpusRetriever, CorpusChunk
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Analogy Agent system prompt
# ---------------------------------------------------------------------------

ANALOGY_AGENT_SYSTEM_PROMPT = """You are the analogy agent. You find structural parallels between the current proof obligation and problems in other mathematical domains.

You receive:
1. The current proof obligation (the specific thing that cannot be proven with current techniques)
2. The obstruction type from the most recent Adversarial Critic output
3. A description of the algebraic structure of the current problem
4. Corpus passages retrieved by structural query

Your task:
1. Identify the most structurally similar problem from a DIFFERENT domain in the corpus passages
2. Describe what technique was used to resolve that problem
3. Propose whether the same move might work here, and HOW precisely

You are looking for STRUCTURAL ISOMORPHISMS, not topical similarity.
  - A problem about counting edges in a graph and a problem about composing functions might be the same problem under the surface
  - A problem about counting lattice points and a problem about convex optimization might share the same duality structure
  - A problem about circuit depth and a problem about proof complexity share the same monotone structure

Do NOT propose techniques that are in the Generator's dead_ends list.
Do NOT propose analogies to the exact same domain the Generator is currently working in.

Output as structured JSON:
{
  "analogous_domain": "<domain name>",
  "analogous_problem": "<description of the structurally similar problem>",
  "structural_isomorphism": "<precisely what algebraic structure is shared>",
  "technique_used_there": "<what technique resolved the analogous problem>",
  "proposed_transfer": "<how to apply that technique here>",
  "confidence": 0.0-1.0,
  "corpus_source": "<which retrieved passage was most relevant>"
}
"""

STRUCTURE_DESCRIPTION_PROMPT = """You are helping formulate a structural query for a mathematical analogy search.

The current proof obligation is:
{proof_obligation}

The obstruction type is: {obstruction_type}

Describe the ALGEBRAIC STRUCTURE of this problem in 2-3 sentences using only abstract mathematical language.
Do NOT use the specific terminology of the problem's surface domain.
Use mathematical structure words: bilinear map, tensor product, rank, composition, linear operator, group action, morphism, fiber bundle, etc.

Your description should be so abstract that a mathematician working in a completely different field could recognize the structure.

Output ONLY the structural description, nothing else.
"""


@dataclass
class AnalogyResult:
    """Output of the Analogy Agent."""
    analogous_domain: str = ""
    analogous_problem: str = ""
    structural_isomorphism: str = ""
    technique_used_there: str = ""
    proposed_transfer: str = ""
    confidence: float = 0.0
    corpus_source: str = ""
    structural_query_used: str = ""
    retrieved_passages: List[Dict] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analogous_domain": self.analogous_domain,
            "analogous_problem": self.analogous_problem,
            "structural_isomorphism": self.structural_isomorphism,
            "technique_used_there": self.technique_used_there,
            "proposed_transfer": self.proposed_transfer,
            "confidence": self.confidence,
            "corpus_source": self.corpus_source,
            "structural_query_used": self.structural_query_used,
        }

    def to_generator_injection(self) -> str:
        """Format for injection into the Generator's context."""
        if not self.analogous_domain:
            return "[Analogy Agent: no relevant analogy found in corpus]"
        return (
            f"=== CROSS-DOMAIN ANALOGY ===\n"
            f"In the domain of [{self.analogous_domain}], a structurally similar "
            f"obstruction was resolved by:\n"
            f"  Technique: {self.technique_used_there}\n"
            f"  Analogous problem: {self.analogous_problem}\n"
            f"  Structural isomorphism: {self.structural_isomorphism}\n\n"
            f"PROPOSED TRANSFER:\n{self.proposed_transfer}\n"
            f"(confidence: {self.confidence:.0%})\n"
            f"=== END ANALOGY ===\n"
            f"\nConsider whether a corresponding move is available in the current setting. "
            f"If the structural isomorphism holds, the same move should work here."
        )

    @property
    def is_useful(self) -> bool:
        return bool(self.analogous_domain) and self.confidence > 0.3


class AnalogyAgent:
    """
    Analogy Agent: finds cross-domain structural parallels.

    EXPERIMENTAL VALIDITY NOTE:
    The agent queries the corpus using an ABSTRACT STRUCTURAL DESCRIPTION
    generated from the current proof obligation, NOT using the technique
    name or result name directly. This prevents the agent from encoding
    prior knowledge of the target result.
    """

    def __init__(self, corpus_retriever: CorpusRetriever) -> None:
        self.corpus_retriever = corpus_retriever

    def query(
        self,
        current_obligation: str,
        obstruction_type: str,
        dead_ends: Optional[List[str]] = None,
        round_num: int = 1,
    ) -> AnalogyResult:
        """
        Find a cross-domain analogy for the current proof obligation.

        Args:
            current_obligation:  The specific thing that cannot be proven.
            obstruction_type:    e.g. "algebraic_obstruction", "complexity_barrier"
            dead_ends:          List of technique descriptions already tried.
            round_num:           Current round number.

        Returns:
            AnalogyResult with the best structural analogy found.
        """
        from utils.model_loader import get_qwen3
        qwen3 = get_qwen3()

        dead_ends = dead_ends or []

        log.info(f"Analogy agent: querying for {obstruction_type} obstruction")

        # Step 1: Generate structural description of the current problem
        structural_query = self._generate_structural_query(
            qwen3, current_obligation, obstruction_type
        )
        log.debug(f"Structural query: {structural_query[:100]}...")

        # Step 2: Retrieve corpus passages by structural similarity
        # NOTE: query is by STRUCTURE, not by topic or technique name
        passages = self.corpus_retriever.retrieve(
            query=structural_query,
            top_k=8,
        )

        if not passages:
            log.info("Analogy agent: no corpus passages found.")
            return AnalogyResult(structural_query_used=structural_query)

        # Step 3: Ask Qwen3 to identify the best analogy
        result = self._identify_analogy(
            qwen3,
            current_obligation=current_obligation,
            obstruction_type=obstruction_type,
            passages=passages,
            dead_ends=dead_ends,
            structural_query=structural_query,
            round_num=round_num,
        )
        result.structural_query_used = structural_query

        log.info(
            f"Analogy agent: found analogy to [{result.analogous_domain}] "
            f"with confidence {result.confidence:.0%}"
        )
        return result

    # -----------------------------------------------------------------------
    # PRIVATE
    # -----------------------------------------------------------------------

    @staticmethod
    def _generate_structural_query(qwen3, obligation: str, obstruction_type: str) -> str:
        """
        Ask Qwen3 to describe the algebraic structure of the current problem
        without domain-specific terminology.
        This query is used for corpus retrieval — it must not encode knowledge
        of the target result.
        """
        prompt = STRUCTURE_DESCRIPTION_PROMPT.format(
            proof_obligation=obligation,
            obstruction_type=obstruction_type,
        )
        messages = [
            {
                "role": "system",
                "content": "You describe mathematical structures in abstract terms."
            },
            {"role": "user", "content": prompt},
        ]
        description = qwen3.generate(
            messages,
            max_tokens=200,
            temperature=0.3,
        )
        return description.strip()

    @staticmethod
    def _identify_analogy(
        qwen3,
        current_obligation: str,
        obstruction_type: str,
        passages: List[CorpusChunk],
        dead_ends: List[str],
        structural_query: str,
        round_num: int,
    ) -> AnalogyResult:
        """Ask Qwen3 to identify the best cross-domain analogy from retrieved passages."""
        # Format retrieved passages
        passages_text = "\n\n".join(
            f"[Source: {p.source_file}, chunk {p.chunk_index}]\n{p.text[:400]}"
            for p in passages[:6]  # top 6 passages
        )

        dead_ends_text = ""
        if dead_ends:
            dead_ends_text = (
                "\nDO NOT propose any of these techniques (already tried):\n"
                + "\n".join(f"  - {d}" for d in dead_ends[:5])
            )

        messages = [
            {"role": "system", "content": ANALOGY_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ROUND: {round_num}\n\n"
                    f"CURRENT PROOF OBLIGATION:\n{current_obligation}\n\n"
                    f"OBSTRUCTION TYPE: {obstruction_type}\n\n"
                    f"ABSTRACT STRUCTURAL DESCRIPTION:\n{structural_query}\n\n"
                    f"CORPUS PASSAGES RETRIEVED BY STRUCTURAL SIMILARITY:\n{passages_text}\n"
                    f"{dead_ends_text}\n\n"
                    f"Find the best cross-domain structural analogy. Output valid JSON."
                ),
            },
        ]

        raw = qwen3.generate(
            messages,
            max_tokens=config.CRITIC_MAX_TOKENS,
            temperature=0.5,
        )

        # Parse JSON
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json_module.loads(json_match.group())
                return AnalogyResult(
                    analogous_domain=data.get("analogous_domain", ""),
                    analogous_problem=data.get("analogous_problem", ""),
                    structural_isomorphism=data.get("structural_isomorphism", ""),
                    technique_used_there=data.get("technique_used_there", ""),
                    proposed_transfer=data.get("proposed_transfer", ""),
                    confidence=float(data.get("confidence", 0.0)),
                    corpus_source=data.get("corpus_source", ""),
                    retrieved_passages=[p.to_dict() for p in passages[:3]],
                    raw_output=raw,
                )
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                pass

        # Fallback: extract what we can from free text
        return AnalogyResult(
            proposed_transfer=raw[:400],
            retrieved_passages=[p.to_dict() for p in passages[:3]],
            raw_output=raw,
        )
