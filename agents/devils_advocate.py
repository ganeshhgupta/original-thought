"""
agents/devils_advocate.py — Devil's Advocate Agent.

The Devil's Advocate makes the strongest possible case that the current
hypothesis is unnecessary, already known, or incorrectly motivated.

This agent plays a different role than the Adversarial Critic:
  - Adversarial Critic: finds technical flaws in the proof
  - Devil's Advocate: argues the WORK ITSELF is unnecessary

The tension between these two is productive. If the Adversarial Critic
finds no technical flaw but the Devil's Advocate cannot find the result
in the existing literature, the hypothesis is likely genuinely novel AND
technically sound.

The Devil's Advocate is NOT trying to be constructive. It is trying to
show that the work is unnecessary.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Devil's Advocate system prompt
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_SYSTEM_PROMPT = """You are the devil's advocate. You defend the existing consensus in the literature against the current hypothesis.
Your role is to make the strongest possible case that the current hypothesis is unnecessary, already known, or incorrectly motivated.

Specifically, you must:
1. Find the closest known result in the literature to the current hypothesis. If it is essentially the same result, say so explicitly.
2. Argue that the motivation for the hypothesis is flawed. What is the simplest explanation for the phenomenon the hypothesis is trying to explain?
3. Identify the step in the current proof sketch that the existing literature already covers. What is actually new here?
4. If the hypothesis is a generalization of a known result, argue that the known result already answers the question.

Your output must follow this EXACT structure:

CLOSEST_KNOWN_RESULT: [cite the closest result from the corpus or general mathematical knowledge]
NOVELTY_CHALLENGE: [argue that the hypothesis is not novel — it is either already known or trivially follows from known results]
MOTIVATION_CHALLENGE: [argue that the motivation for the hypothesis is flawed or overstated]
WHAT_IS_ACTUALLY_NEW: [if anything — what, if anything, is genuinely new here? Be precise.]
VERDICT: [UNNECESSARY | TRIVIAL | MARGINALLY_NOVEL | GENUINELY_NOVEL]

You are NOT trying to be constructive. You are trying to show that the work is unnecessary.
"""


@dataclass
class DevilsAdvocateResult:
    """Structured output from the Devil's Advocate."""
    closest_known_result: str
    novelty_challenge: str
    motivation_challenge: str
    what_is_actually_new: str
    verdict: str   # UNNECESSARY | TRIVIAL | MARGINALLY_NOVEL | GENUINELY_NOVEL
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closest_known_result": self.closest_known_result,
            "novelty_challenge": self.novelty_challenge,
            "motivation_challenge": self.motivation_challenge,
            "what_is_actually_new": self.what_is_actually_new,
            "verdict": self.verdict,
        }

    def to_context_string(self) -> str:
        return (
            f"[Devil's Advocate — verdict: {self.verdict}]\n"
            f"  Closest known result: {self.closest_known_result}\n"
            f"  Novelty challenge: {self.novelty_challenge[:150]}...\n"
            f"  Motivation challenge: {self.motivation_challenge[:150]}...\n"
            f"  What is actually new: {self.what_is_actually_new[:150]}...\n"
        )

    @property
    def is_blocking(self) -> bool:
        """True if the verdict suggests the work is not worth continuing."""
        return self.verdict in ("UNNECESSARY", "TRIVIAL")


class DevilsAdvocateAgent:
    """Devil's Advocate: challenges the necessity and novelty of the hypothesis."""

    def critique(
        self,
        current_hypothesis: str,
        proof_sketch: str,
        scratchpad_summary: str,
        corpus_context: Optional[str] = None,
        round_num: int = 1,
    ) -> DevilsAdvocateResult:
        """
        Argue against the current hypothesis.

        Args:
            current_hypothesis:  The main conjecture being pursued.
            proof_sketch:        The Generator's current proof sketch.
            scratchpad_summary:  Current scratchpad state.
            corpus_context:      Optional: relevant passages from the corpus
                                 that the agent should use to ground its argument.
            round_num:           Current round number.

        Returns:
            DevilsAdvocateResult with structured argument.
        """
        from utils.model_loader import get_qwen3
        qwen3 = get_qwen3()

        prompt = self._build_prompt(
            current_hypothesis=current_hypothesis,
            proof_sketch=proof_sketch,
            scratchpad_summary=scratchpad_summary,
            corpus_context=corpus_context,
            round_num=round_num,
        )

        messages = [
            {"role": "system", "content": DEVILS_ADVOCATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = qwen3.generate(
            messages,
            max_tokens=config.CRITIC_MAX_TOKENS,
            temperature=config.CRITIC_TEMPERATURE,
        )

        result = self._parse_output(raw)
        log.info(
            f"Devil's Advocate round {round_num}: verdict={result.verdict}"
        )
        return result

    # -----------------------------------------------------------------------
    # PRIVATE
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        current_hypothesis: str,
        proof_sketch: str,
        scratchpad_summary: str,
        corpus_context: Optional[str],
        round_num: int,
    ) -> str:
        parts = [
            f"ROUND: {round_num}\n",
            f"CURRENT HYPOTHESIS:\n{current_hypothesis}\n",
            f"PROOF SKETCH:\n{proof_sketch}\n",
            f"SCRATCHPAD STATE:\n{scratchpad_summary}\n",
        ]
        if corpus_context:
            parts.append(f"RELEVANT CORPUS PASSAGES:\n{corpus_context}\n")
        parts.append(
            "Make the strongest possible case that this hypothesis is unnecessary, "
            "already known, or incorrectly motivated. Follow your system prompt format exactly."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_output(raw: str) -> DevilsAdvocateResult:
        """Parse structured Devil's Advocate output."""
        def _extract(label: str, text: str) -> str:
            pattern = rf'{re.escape(label)}:\s*(.+?)(?=\n[A-Z_]+:|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        verdict_raw = _extract("VERDICT", raw).upper()
        valid_verdicts = {"UNNECESSARY", "TRIVIAL", "MARGINALLY_NOVEL", "GENUINELY_NOVEL"}
        if verdict_raw not in valid_verdicts:
            # Try to find a valid verdict in the string
            for v in valid_verdicts:
                if v in verdict_raw:
                    verdict_raw = v
                    break
            else:
                verdict_raw = "MARGINALLY_NOVEL"

        return DevilsAdvocateResult(
            closest_known_result=_extract("CLOSEST_KNOWN_RESULT", raw),
            novelty_challenge=_extract("NOVELTY_CHALLENGE", raw),
            motivation_challenge=_extract("MOTIVATION_CHALLENGE", raw),
            what_is_actually_new=_extract("WHAT_IS_ACTUALLY_NEW", raw),
            verdict=verdict_raw,
            raw_output=raw,
        )
