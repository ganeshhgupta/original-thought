"""
agents/synthesizer.py — Synthesizer Agent.

The Synthesizer manages the multi-round reasoning process. At the end of
each round it:

  1. TAGS each objection: RESOLVED | OPEN | DEGENERATE
  2. UPDATES the scratchpad: promotes Lean-verified claims to established
  3. CHECKS termination signals (after round 10)
  4. RANKS surviving branches by composite score
  5. DECIDES whether to trigger Constraint Relaxer (every N rounds)
  6. DECIDES whether to trigger Analogy Agent (stall detection)
  7. WRITES a one-paragraph round summary

The Synthesizer is the only agent that has a global view of the session.
It is the one that decides when to redirect and when to terminate.

Output: SynthesisResult (JSON-serializable)
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
# Synthesizer system prompt
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM_PROMPT = """You are the synthesizer of a multi-round mathematical reasoning process.
At the end of each round you receive the Generator's proof sketch, all critic objections, Lean results, and the scratchpad state.

Your tasks:

1. TAG each objection with one of:
   RESOLVED — the Generator directly addressed this objection this round
   OPEN — the objection remains unaddressed
   DEGENERATE — this objection is repeating without adding new information (kill it)

2. IDENTIFY what was established this round: which claims advanced, which stalled.

3. ASSESS the proof attempt: is there genuine progress? What is the strongest surviving branch?

4. WRITE a one-paragraph round summary that a mathematician would find informative.

Output as JSON with this structure:
{
  "objection_tags": {"<objection_type>": "RESOLVED|OPEN|DEGENERATE"},
  "established_this_round": ["<name>", ...],
  "stalled_this_round": ["<name>", ...],
  "strongest_branch": "<description of most promising current direction>",
  "round_summary": "<one paragraph>",
  "trigger_constraint_relaxer": true|false,
  "trigger_analogy_agent": true|false,
  "notes": "<any other observations>"
}
"""


@dataclass
class TerminationSignals:
    """State of the three termination signals."""
    generator_entropy_fires: bool = False
    critic_novelty_fires: bool = False
    lean_progress_fires: bool = False

    @property
    def count_firing(self) -> int:
        return sum([
            self.generator_entropy_fires,
            self.critic_novelty_fires,
            self.lean_progress_fires,
        ])

    @property
    def all_three_fire(self) -> bool:
        return self.count_firing == 3

    @property
    def two_fire(self) -> bool:
        return self.count_firing >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_entropy_fires": self.generator_entropy_fires,
            "critic_novelty_fires": self.critic_novelty_fires,
            "lean_progress_fires": self.lean_progress_fires,
            "count_firing": self.count_firing,
        }


@dataclass
class SynthesisResult:
    """Full output of a Synthesizer pass."""
    round_num: int
    objection_tags: Dict[str, str] = field(default_factory=dict)
    established_this_round: List[str] = field(default_factory=list)
    stalled_this_round: List[str] = field(default_factory=list)
    strongest_branch: str = ""
    round_summary: str = ""
    trigger_constraint_relaxer: bool = False
    trigger_analogy_agent: bool = False
    termination_signals: Optional[TerminationSignals] = None
    notes: str = ""
    raw_output: str = ""

    def lean_obligations_discharged(self) -> int:
        """Count of new Lean obligations discharged this round (proxy: established count)."""
        return len(self.established_this_round)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "objection_tags": self.objection_tags,
            "established_this_round": self.established_this_round,
            "stalled_this_round": self.stalled_this_round,
            "strongest_branch": self.strongest_branch,
            "round_summary": self.round_summary,
            "trigger_constraint_relaxer": self.trigger_constraint_relaxer,
            "trigger_analogy_agent": self.trigger_analogy_agent,
            "termination_signals": self.termination_signals.to_dict() if self.termination_signals else None,
            "notes": self.notes,
        }


class SynthesizerAgent:
    """
    Synthesizer: manages the round structure and termination signals.

    Maintains per-session history for signal computation.
    """

    def __init__(self) -> None:
        # Per-session history for signal computation
        self._generator_outputs: List[str] = []   # last N generator outputs
        self._critic_tags: List[str] = []          # OPEN/RESOLVED/DEGENERATE per round
        self._lean_progress: List[int] = []         # obligations discharged per round

    def process(
        self,
        generator_output,
        lean_results: List[Any],
        adversarial_result,
        devils_result,
        deepseek_result: Dict[str, Any],
        scratchpad: Scratchpad,
        round_num: int,
        session,
    ) -> SynthesisResult:
        """
        Process one complete round.

        Args:
            generator_output:    GeneratorOutput from the Generator agent.
            lean_results:        List of LeanResult objects.
            adversarial_result:  AdversarialResult from the Adversarial Critic.
            devils_result:       DevilsAdvocateResult from the Devil's Advocate.
            deepseek_result:     Dict from DeepSeek-Prover evaluation.
            scratchpad:          Current scratchpad state.
            round_num:           Current round number.
            session:             Session state object.

        Returns:
            SynthesisResult with all decisions for this round.
        """
        from utils.model_loader import get_qwen3
        qwen3 = get_qwen3()

        # Track generator output for entropy signal
        self._generator_outputs.append(generator_output.raw_output)

        # Track Lean progress
        lean_discharged = sum(
            r.obligations_discharged for r in lean_results if r.success
        )
        self._lean_progress.append(lean_discharged)

        # Track critic novelty
        if adversarial_result.is_exhausted:
            self._critic_tags.append("DEGENERATE")
        else:
            self._critic_tags.append("OPEN")  # will be updated after synthesis

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(
            generator_output=generator_output,
            lean_results=lean_results,
            adversarial_result=adversarial_result,
            devils_result=devils_result,
            deepseek_result=deepseek_result,
            scratchpad=scratchpad,
            round_num=round_num,
        )

        messages = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = qwen3.generate(
            messages,
            max_tokens=config.CRITIC_MAX_TOKENS,
            temperature=0.2,  # more deterministic for synthesis
        )

        result = self._parse_synthesis(raw, round_num)

        # Compute termination signals (only after round 10)
        if round_num > 10:
            result.termination_signals = self._compute_termination_signals(round_num)
        else:
            result.termination_signals = TerminationSignals()

        # Apply scratchpad promotions
        for name in result.established_this_round:
            if name in scratchpad.conjectures:
                lean_proof = self._find_lean_proof(name, lean_results)
                if lean_proof:
                    scratchpad.promote_to_established(name, lean_proof)

        # Check round-based triggers
        if round_num % config.CONSTRAINT_RELAXATION_INTERVAL == 0:
            result.trigger_constraint_relaxer = True

        # Check analogy trigger (stall detection)
        stall_rounds = self._count_stall_rounds()
        if stall_rounds >= config.ANALOGY_ACTIVATION_ROUND:
            result.trigger_analogy_agent = True

        log.info(
            f"Synthesizer round {round_num}: "
            f"established={result.established_this_round}, "
            f"lean_discharged={lean_discharged}, "
            f"signals={result.termination_signals.count_firing if result.termination_signals else 0}"
        )

        return result

    # -----------------------------------------------------------------------
    # TERMINATION SIGNALS
    # -----------------------------------------------------------------------

    def _compute_termination_signals(self, round_num: int) -> TerminationSignals:
        """
        Compute all three termination signals.

        Signal 1 (Generator Entropy): average pairwise cosine similarity of
          last 5 generator outputs > DEGENERATE_SIMILARITY_THRESHOLD.

        Signal 2 (Critic Novelty Rate): fraction of DEGENERATE critic tags
          in last 5 rounds > (1 - CRITIC_NOVELTY_MIN_RATE).

        Signal 3 (Lean Progress Rate): zero new obligations discharged in
          each of the last LEAN_FLAT_ROUNDS rounds.
        """
        signals = TerminationSignals()

        # Signal 1: Generator entropy
        if len(self._generator_outputs) >= 5:
            try:
                from utils.model_loader import embed
                recent = self._generator_outputs[-5:]
                embeddings = embed(recent)
                avg_sim = self._avg_pairwise_cosine(embeddings)
                signals.generator_entropy_fires = avg_sim > config.DEGENERATE_SIMILARITY_THRESHOLD
                log.debug(f"Generator entropy signal: avg_sim={avg_sim:.3f}, fires={signals.generator_entropy_fires}")
            except Exception as e:
                log.warning(f"Generator entropy signal computation failed: {e}")

        # Signal 2: Critic novelty rate
        if len(self._critic_tags) >= 5:
            recent_tags = self._critic_tags[-5:]
            degenerate_rate = sum(1 for t in recent_tags if t == "DEGENERATE") / len(recent_tags)
            signals.critic_novelty_fires = degenerate_rate > (1.0 - config.CRITIC_NOVELTY_MIN_RATE)
            log.debug(f"Critic novelty signal: degenerate_rate={degenerate_rate:.3f}, fires={signals.critic_novelty_fires}")

        # Signal 3: Lean progress rate
        if len(self._lean_progress) >= config.LEAN_FLAT_ROUNDS:
            recent_lean = self._lean_progress[-config.LEAN_FLAT_ROUNDS:]
            signals.lean_progress_fires = all(p == 0 for p in recent_lean)
            log.debug(f"Lean progress signal: recent={recent_lean}, fires={signals.lean_progress_fires}")

        return signals

    def _count_stall_rounds(self) -> int:
        """Count consecutive rounds with no Lean progress at end of history."""
        count = 0
        for p in reversed(self._lean_progress):
            if p == 0:
                count += 1
            else:
                break
        return count

    # -----------------------------------------------------------------------
    # PRIVATE: PROMPT AND PARSING
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_synthesis_prompt(
        generator_output,
        lean_results: List[Any],
        adversarial_result,
        devils_result,
        deepseek_result: Dict,
        scratchpad: Scratchpad,
        round_num: int,
    ) -> str:
        lean_summary = "\n".join(
            f"  - {r.lean_code[:60]}... "
            f"{'SUCCESS' if r.success else 'FAIL'} "
            f"({r.obligations_discharged} discharged, {r.obligations_remaining} remaining)"
            for r in lean_results
        ) or "  (no Lean calls this round)"

        return (
            f"ROUND: {round_num}\n\n"
            f"GENERATOR OUTPUT (final statement):\n"
            f"{generator_output.final_statement}\n\n"
            f"LEAN VERIFICATION RESULTS:\n{lean_summary}\n\n"
            f"ADVERSARIAL CRITIC:\n{adversarial_result.to_context_string()}\n\n"
            f"DEVIL'S ADVOCATE:\n{devils_result.to_context_string()}\n\n"
            f"DEEPSEEK-PROVER VERDICT:\n"
            f"  valid={deepseek_result.get('valid')}, "
            f"confidence={deepseek_result.get('confidence', 0):.2f}\n\n"
            f"SCRATCHPAD STATE:\n{scratchpad.context_summary()}\n\n"
            f"Synthesize this round's output. Tag each objection, identify what was "
            f"established, write a round summary, and output valid JSON."
        )

    @staticmethod
    def _parse_synthesis(raw: str, round_num: int) -> SynthesisResult:
        """Parse JSON synthesis output with fallback."""
        # Extract JSON block
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return SynthesisResult(
                    round_num=round_num,
                    objection_tags=data.get("objection_tags", {}),
                    established_this_round=data.get("established_this_round", []),
                    stalled_this_round=data.get("stalled_this_round", []),
                    strongest_branch=data.get("strongest_branch", ""),
                    round_summary=data.get("round_summary", ""),
                    trigger_constraint_relaxer=data.get("trigger_constraint_relaxer", False),
                    trigger_analogy_agent=data.get("trigger_analogy_agent", False),
                    notes=data.get("notes", ""),
                    raw_output=raw,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: extract round summary from free text
        summary_match = re.search(
            r'"round_summary"\s*:\s*"([^"]+)"', raw
        )
        summary = summary_match.group(1) if summary_match else raw[:300]

        return SynthesisResult(
            round_num=round_num,
            round_summary=summary,
            raw_output=raw,
        )

    @staticmethod
    def _find_lean_proof(name: str, lean_results: List[Any]) -> Optional[str]:
        """Find a Lean proof string for a named claim in the results list."""
        for result in lean_results:
            if result.success and name in result.lean_code:
                return result.lean_code
        return None

    @staticmethod
    def _avg_pairwise_cosine(embeddings: List[List[float]]) -> float:
        """Compute average pairwise cosine similarity of a list of embeddings."""
        from utils.model_loader import cosine_similarity
        n = len(embeddings)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += cosine_similarity(embeddings[i], embeddings[j])
                count += 1
        return total / count if count > 0 else 0.0
