"""
agents/adversarial_critic.py — Adversarial Critic Agent.

The Adversarial Critic finds the WEAKEST POINT in the current proof sketch.
It must produce a different objection than any raised in the last
CRITIC_NOVELTY_WINDOW rounds. If it cannot find a new objection, it outputs
EXHAUSTED, which is a signal that the proof territory has been thoroughly
explored.

The critic's output is highly structured — the obstruction type, the
specific assumption being challenged, and the targeted proof step are all
extracted and stored in the failure memory database for future sessions.

IMPORTANT: This agent never references the target result. Its system prompt
contains no steering toward or away from any specific conclusion. It is a
pure proof-checker with no knowledge of what a "correct" answer would be.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Adversarial Critic system prompt
# ---------------------------------------------------------------------------

ADVERSARIAL_CRITIC_SYSTEM_PROMPT = """You are a mathematical critic. You receive a proof sketch and must find its weakest point.

Your output must follow this EXACT structure (use these exact headers):

OBJECTION_TYPE: [logical_gap | missing_lemma | contradicts_corpus | algebraic_obstruction | dimensional_mismatch | complexity_barrier | other]
ASSUMPTION_CHALLENGED: [name the specific assumption being relied upon]
PROOF_STEP_TARGETED: [which step number fails, or 0 if the entire framing is flawed]
FORMAL_OBJECTION: [state precisely what is missing or wrong — be specific about which mathematical object fails and why]
COUNTEREXAMPLE_SKETCH: [if you can construct a counterexample, sketch it with explicit parameters; if not, write NONE]
WHAT_WOULD_FIX_IT: [what lemma or modification would patch this specific gap — be concrete]

RULES:
1. You must produce a DIFFERENT objection than any objection raised in the previous rounds (listed below).
2. If you cannot find a genuinely new objection, output EXHAUSTED on its own line.
3. Do not raise vague objections. Every objection must target a specific step or specific assumption.
4. Do not object to things already in the ESTABLISHED section of the scratchpad.
5. Priority order for objections: algebraic_obstruction > logical_gap > missing_lemma > complexity_barrier > dimensional_mismatch > other
"""


@dataclass
class AdversarialResult:
    """Structured output from the Adversarial Critic."""
    objection_type: str      # one of the enum values or "exhausted"
    assumption_challenged: str
    proof_step_targeted: int
    formal_objection: str
    counterexample_sketch: Optional[str]
    what_would_fix_it: str
    is_exhausted: bool = False
    raw_output: str = ""
    tag: str = "OPEN"       # OPEN | RESOLVED | DEGENERATE (set by Synthesizer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objection_type": self.objection_type,
            "assumption_challenged": self.assumption_challenged,
            "proof_step_targeted": self.proof_step_targeted,
            "formal_objection": self.formal_objection,
            "counterexample_sketch": self.counterexample_sketch,
            "what_would_fix_it": self.what_would_fix_it,
            "is_exhausted": self.is_exhausted,
            "tag": self.tag,
        }

    def to_context_string(self) -> str:
        """Format for injection into the Generator's next round context."""
        if self.is_exhausted:
            return "[Adversarial Critic: EXHAUSTED — no new objections found]"
        return (
            f"[Adversarial Critic]\n"
            f"  Objection type: {self.objection_type}\n"
            f"  Assumption challenged: {self.assumption_challenged}\n"
            f"  Step targeted: {self.proof_step_targeted}\n"
            f"  Objection: {self.formal_objection}\n"
            f"  Would fix it: {self.what_would_fix_it}\n"
            + (f"  Counterexample: {self.counterexample_sketch}\n"
               if self.counterexample_sketch and self.counterexample_sketch != "NONE"
               else "")
        )


class AdversarialCriticAgent:
    """
    Adversarial Critic: finds the weakest point in the current proof sketch.

    Novelty tracking: maintains a window of past objections and forces the
    critic to find a different one. When the critic outputs EXHAUSTED, the
    Synthesizer counts this in the critic novelty rate signal.
    """

    def __init__(self) -> None:
        pass  # Stateless — uses Qwen3 on demand

    def critique(
        self,
        proof_sketch: str,
        scratchpad_summary: str,
        previous_objections: List[Dict[str, Any]],
        round_num: int = 1,
    ) -> AdversarialResult:
        """
        Critique the current proof sketch.

        Args:
            proof_sketch:         The Generator's current proof sketch.
            scratchpad_summary:   Current scratchpad state.
            previous_objections:  List of past objection dicts from recent rounds.
            round_num:            Current round number.

        Returns:
            AdversarialResult with structured objection.
        """
        from utils.model_loader import get_qwen3
        qwen3 = get_qwen3()

        # Format previous objections for injection
        prev_obj_text = self._format_previous_objections(
            previous_objections[-config.CRITIC_NOVELTY_WINDOW:]
        )

        messages = [
            {"role": "system", "content": ADVERSARIAL_CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_critique_prompt(
                    proof_sketch=proof_sketch,
                    scratchpad_summary=scratchpad_summary,
                    prev_obj_text=prev_obj_text,
                    round_num=round_num,
                ),
            },
        ]

        raw = qwen3.generate(
            messages,
            max_tokens=config.CRITIC_MAX_TOKENS,
            temperature=config.CRITIC_TEMPERATURE,
        )

        result = self._parse_critic_output(raw)
        log.info(
            f"Adversarial Critic round {round_num}: "
            + ("EXHAUSTED" if result.is_exhausted else
               f"{result.objection_type} @ step {result.proof_step_targeted}")
        )
        return result

    # -----------------------------------------------------------------------
    # PRIVATE: PROMPT AND PARSING
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_critique_prompt(
        proof_sketch: str,
        scratchpad_summary: str,
        prev_obj_text: str,
        round_num: int,
    ) -> str:
        return (
            f"ROUND: {round_num}\n\n"
            f"CURRENT PROOF SKETCH:\n{proof_sketch}\n\n"
            f"SCRATCHPAD STATE:\n{scratchpad_summary}\n\n"
            + (f"PREVIOUS OBJECTIONS (you MUST find a different one):\n{prev_obj_text}\n\n"
               if prev_obj_text else "")
            + "Find the weakest point in the proof sketch above. "
              "Follow the exact output format specified in your system prompt."
        )

    @staticmethod
    def _format_previous_objections(objections: List[Dict]) -> str:
        if not objections:
            return ""
        lines = []
        for i, obj in enumerate(objections, 1):
            lines.append(
                f"  {i}. [{obj.get('objection_type', '?')}] "
                f"{obj.get('formal_objection', '')[:100]}..."
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_critic_output(raw: str) -> AdversarialResult:
        """Parse the structured critic output."""
        raw_stripped = raw.strip()

        # Check for EXHAUSTED
        if raw_stripped.upper().startswith("EXHAUSTED") or "EXHAUSTED" in raw_stripped[:50]:
            return AdversarialResult(
                objection_type="exhausted",
                assumption_challenged="",
                proof_step_targeted=0,
                formal_objection="No new objection found",
                counterexample_sketch=None,
                what_would_fix_it="",
                is_exhausted=True,
                raw_output=raw,
                tag="DEGENERATE",
            )

        def _extract(label: str, text: str) -> str:
            pattern = rf'{re.escape(label)}:\s*(.+?)(?=\n[A-Z_]+:|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        objection_type = _extract("OBJECTION_TYPE", raw_stripped).lower()
        # Normalize objection type to valid enum values
        valid_types = {
            "logical_gap", "missing_lemma", "contradicts_corpus",
            "algebraic_obstruction", "dimensional_mismatch", "complexity_barrier", "other"
        }
        if objection_type not in valid_types:
            # Try to match a valid type from the extracted string
            for vt in valid_types:
                if vt in objection_type:
                    objection_type = vt
                    break
            else:
                objection_type = "other"

        step_str = _extract("PROOF_STEP_TARGETED", raw_stripped)
        try:
            proof_step = int(re.search(r'\d+', step_str).group()) if re.search(r'\d+', step_str) else 0
        except (AttributeError, ValueError):
            proof_step = 0

        return AdversarialResult(
            objection_type=objection_type,
            assumption_challenged=_extract("ASSUMPTION_CHALLENGED", raw_stripped),
            proof_step_targeted=proof_step,
            formal_objection=_extract("FORMAL_OBJECTION", raw_stripped),
            counterexample_sketch=_extract("COUNTEREXAMPLE_SKETCH", raw_stripped) or None,
            what_would_fix_it=_extract("WHAT_WOULD_FIX_IT", raw_stripped),
            is_exhausted=False,
            raw_output=raw,
            tag="OPEN",
        )
