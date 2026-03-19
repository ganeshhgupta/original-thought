"""
agents/constraint_relaxer.py — Constraint Relaxation Agent.

The Constraint Relaxer examines the current scratchpad's ASSUMPTIONS section
and proposes variants: weakened, negated, or replaced assumptions.

This operationalizes the core human discovery mechanism:
  "Is this assumption actually necessary?"

Major mathematical discoveries that followed from this mechanism:
  - Non-Euclidean geometry: what if the parallel postulate is not needed?
  - Special relativity: what if simultaneity is not absolute?
  - Non-standard analysis: what if infinitesimals are consistent after all?
  - Linear logic: what if we track how many times each resource is used?

The Constraint Relaxer runs every CONSTRAINT_RELAXATION_INTERVAL rounds and
whenever the redirect protocol fires. It generates new branch specifications
that the Generator can explore in subsequent rounds.

Each generated variant is a new potential branch for the Generator. The
Synthesizer decides which branches to pursue.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
from memory.scratchpad import Scratchpad
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constraint Relaxer system prompt
# ---------------------------------------------------------------------------

CONSTRAINT_RELAXER_SYSTEM_PROMPT = """You are the constraint relaxation agent. You examine the current proof attempt's assumption list and propose variants.
You receive the current scratchpad's ASSUMPTIONS section.

For EACH assumption, propose THREE variants:

1. WEAKENED: What if this assumption held only approximately, or only for a subclass of cases, or only in the limit?
   - State the weakened version precisely
   - Assess: is the weakened version still sufficient to support the main proof goal?

2. NEGATED: What if this assumption were false?
   - Does the impossibility proof break if this assumption is removed?
   - Does removing it open a new approach?

3. REPLACED: What other assumption could substitute for this one while preserving the key properties?
   - Is there a structurally different but mathematically equivalent replacement?
   - Does the replacement connect the problem to a different area of mathematics?

For EACH variant, estimate:
  - Whether it changes the proof goal's status (easier/harder/equivalent)
  - Whether it introduces a tractable new proof obligation
  - Whether it connects to a different area of mathematics

Output as JSON:
{
  "variants": [
    {
      "original_assumption": "<name>",
      "original_statement": "<statement>",
      "weakened": {
        "statement": "<weakened version>",
        "sufficient": true|false|"unknown",
        "new_obligation": "<what would need to be proven>",
        "connects_to": "<mathematical area, if any>"
      },
      "negated": {
        "statement": "<negation>",
        "breaks_impossibility": true|false|"unknown",
        "opens_approach": "<description of new approach if any>"
      },
      "replaced": {
        "statement": "<replacement>",
        "equivalent": true|false|"unknown",
        "connects_to": "<mathematical area>",
        "tractability": "<assessment>"
      }
    }
  ],
  "most_promising_relaxation": "<name of assumption + type of relaxation>",
  "new_branch_specification": "<precise description of the most promising new direction to explore>"
}
"""


@dataclass
class AssumptionVariant:
    """A single set of variants for one assumption."""
    original_assumption: str
    original_statement: str
    weakened: Dict[str, Any] = field(default_factory=dict)
    negated: Dict[str, Any] = field(default_factory=dict)
    replaced: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_assumption": self.original_assumption,
            "original_statement": self.original_statement,
            "weakened": self.weakened,
            "negated": self.negated,
            "replaced": self.replaced,
        }

    def to_branch_prompt(self) -> str:
        """Format as a branch specification for the Generator."""
        lines = [
            f"CONSTRAINT RELAXATION — Assumption: {self.original_assumption}",
            f"  Original: {self.original_statement}",
        ]
        if self.weakened.get("statement"):
            lines.append(f"  Weakened: {self.weakened['statement']}")
            if self.weakened.get("new_obligation"):
                lines.append(f"    New obligation: {self.weakened['new_obligation']}")
        if self.negated.get("breaks_impossibility"):
            lines.append(
                f"  Negation: {self.negated.get('statement', 'unknown')} "
                f"[breaks impossibility: {self.negated.get('breaks_impossibility')}]"
            )
        if self.replaced.get("statement"):
            lines.append(
                f"  Replacement: {self.replaced['statement']} "
                f"[connects to: {self.replaced.get('connects_to', 'unknown')}]"
            )
        return "\n".join(lines)


@dataclass
class ConstraintRelaxationResult:
    """Output of a Constraint Relaxer pass."""
    variants: List[AssumptionVariant] = field(default_factory=list)
    most_promising_relaxation: str = ""
    new_branch_specification: str = ""
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variants": [v.to_dict() for v in self.variants],
            "most_promising_relaxation": self.most_promising_relaxation,
            "new_branch_specification": self.new_branch_specification,
        }

    def to_generator_injection(self) -> str:
        """Format for injection into the Generator's next round."""
        lines = [
            "=== CONSTRAINT RELAXATION ANALYSIS ===",
            "The following assumptions have been analyzed. "
            "Consider whether any of these relaxations opens a new proof path:\n",
        ]
        for v in self.variants[:3]:  # top 3 to avoid context overload
            lines.append(v.to_branch_prompt())
            lines.append("")

        if self.new_branch_specification:
            lines.append(
                f"MOST PROMISING NEW DIRECTION:\n{self.new_branch_specification}"
            )
        lines.append("=== END CONSTRAINT RELAXATION ===")
        return "\n".join(lines)


class ConstraintRelaxerAgent:
    """
    Constraint Relaxation Agent: proposes weakened/negated/replaced assumptions.

    Prioritizes assumptions that are:
      1. Most load-bearing (cited most in proof sketches)
      2. Not yet challenged in this session
      3. Most similar to assumptions in known dead ends (to avoid them)
    """

    def generate_variants(
        self,
        scratchpad: Scratchpad,
        focus_assumption: Optional[str] = None,
    ) -> ConstraintRelaxationResult:
        """
        Generate relaxation variants for assumptions in the scratchpad.

        Args:
            scratchpad:        Current working memory state.
            focus_assumption:  If provided, focus analysis on this specific assumption.

        Returns:
            ConstraintRelaxationResult with all variants and new branch specification.
        """
        from utils.model_loader import get_qwen3
        qwen3 = get_qwen3()

        if not scratchpad.assumptions:
            log.info("Constraint relaxer: no assumptions to relax.")
            return ConstraintRelaxationResult(
                new_branch_specification=(
                    "No assumptions currently in scratchpad. "
                    "Consider writing explicit assumptions before calling this agent."
                )
            )

        # Prioritize assumptions
        assumptions_to_analyze = self._prioritize_assumptions(
            scratchpad, focus_assumption
        )

        prompt = self._build_prompt(
            assumptions=assumptions_to_analyze,
            established=scratchpad.established,
            conjectures=scratchpad.conjectures,
            dead_ends=scratchpad.dead_ends,
        )

        messages = [
            {"role": "system", "content": CONSTRAINT_RELAXER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = qwen3.generate(
            messages,
            max_tokens=config.CRITIC_MAX_TOKENS,
            temperature=0.4,  # slightly more creative for generating variants
        )

        result = self._parse_output(raw, assumptions_to_analyze)

        log.info(
            f"Constraint relaxer: {len(result.variants)} variants generated. "
            f"Most promising: {result.most_promising_relaxation}"
        )
        return result

    # -----------------------------------------------------------------------
    # PRIVATE
    # -----------------------------------------------------------------------

    @staticmethod
    def _prioritize_assumptions(
        scratchpad: Scratchpad,
        focus_assumption: Optional[str],
    ) -> List[Tuple[str, str]]:
        """
        Return (name, statement) pairs for assumptions to analyze,
        ordered by priority.
        """
        if focus_assumption and focus_assumption in scratchpad.assumptions:
            # Start with the focused assumption
            focused = [(focus_assumption, scratchpad.assumptions[focus_assumption]["statement"])]
            rest = [
                (name, entry["statement"])
                for name, entry in scratchpad.assumptions.items()
                if name != focus_assumption
            ]
            return focused + rest[:4]  # at most 5 total

        # Default: prioritize load-bearing and unchallenged
        most_lb = scratchpad.most_load_bearing_assumption()
        unchallenged = scratchpad.unchallenged_assumptions()

        result = []
        if most_lb:
            result.append(most_lb)
        for name, stmt in unchallenged:
            if not result or name != result[0][0]:
                result.append((name, stmt))

        # Fill with any remaining
        for name, entry in scratchpad.assumptions.items():
            if not any(n == name for n, _ in result):
                result.append((name, entry["statement"]))

        return result[:5]  # at most 5 assumptions analyzed per pass

    @staticmethod
    def _build_prompt(
        assumptions: List[Tuple[str, str]],
        established: Dict,
        conjectures: Dict,
        dead_ends: List,
    ) -> str:
        parts = ["ASSUMPTIONS TO ANALYZE:\n"]
        for name, stmt in assumptions:
            parts.append(f"  [{name}]: {stmt}\n")

        if established:
            parts.append("\nESTABLISHED RESULTS (context):\n")
            for name, entry in list(established.items())[:5]:
                parts.append(f"  [{name}]: {entry['statement']}\n")

        if conjectures:
            parts.append("\nCURRENT PROOF GOALS:\n")
            for name, entry in list(conjectures.items())[:3]:
                parts.append(f"  [{name}]: {entry['statement']}\n")

        if dead_ends:
            parts.append("\nKNOWN DEAD ENDS (avoid generating variants that lead here):\n")
            for de in dead_ends[-3:]:
                parts.append(f"  [{de['obstruction_type']}]: {de['description'][:80]}\n")

        parts.append(
            "\nFor each assumption, generate WEAKENED, NEGATED, and REPLACED variants. "
            "Output as valid JSON per your system prompt."
        )
        return "".join(parts)

    @staticmethod
    def _parse_output(
        raw: str,
        assumptions: List[Tuple[str, str]],
    ) -> ConstraintRelaxationResult:
        """Parse JSON output from the Constraint Relaxer."""
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                variants = []
                for v_data in data.get("variants", []):
                    variants.append(AssumptionVariant(
                        original_assumption=v_data.get("original_assumption", ""),
                        original_statement=v_data.get("original_statement", ""),
                        weakened=v_data.get("weakened", {}),
                        negated=v_data.get("negated", {}),
                        replaced=v_data.get("replaced", {}),
                    ))
                return ConstraintRelaxationResult(
                    variants=variants,
                    most_promising_relaxation=data.get("most_promising_relaxation", ""),
                    new_branch_specification=data.get("new_branch_specification", ""),
                    raw_output=raw,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: create minimal variants from the assumptions we were given
        variants = [
            AssumptionVariant(
                original_assumption=name,
                original_statement=stmt,
            )
            for name, stmt in assumptions
        ]
        return ConstraintRelaxationResult(
            variants=variants,
            new_branch_specification=raw[:300],
            raw_output=raw,
        )
