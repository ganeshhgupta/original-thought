"""
main.py — Entry point for the NOVA Reasoning System.

Usage:
  python main.py --problem ./problem_prompt.txt [--rounds 30] [--session-id auto] [--resume SESSION_ID]

The problem prompt is the ONLY external steering mechanism. The system reads
it once at startup. Nothing else references the target answer.

Session lifecycle:
  1. Load failure memory from past sessions
  2. Sample framing variant via Thompson sampling
  3. DPP-sample corpus ordering for this session
  4. Initialize scratchpad
  5. Main round loop (see pseudocode in spec)
  6. Score surviving conjectures
  7. Write outputs (results.json + report.md)
  8. Update failure memory and Thompson sampling stats

Resumption: if --resume SESSION_ID is passed, the system loads the checkpoint
from that session and continues from the last completed round.
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from agents.generator import GeneratorAgent, GeneratorOutput
from agents.adversarial_critic import AdversarialCriticAgent, AdversarialResult
from agents.devils_advocate import DevilsAdvocateAgent
from agents.synthesizer import SynthesizerAgent, TerminationSignals, SynthesisResult
from agents.constraint_relaxer import ConstraintRelaxerAgent
from agents.analogy_agent import AnalogyAgent
from memory.failure_store import FailureStore, FailureEntry
from memory.scratchpad import Scratchpad
from scoring.scorer import ConjectureScorer, ConjectureScore
from tools.lean_tool import LeanTool
from tools.z3_tool import Z3Tool
from tools.sympy_tool import SymPyTool
from tools.corpus_retriever import CorpusRetriever
from tools.dpp_sampler import DPPSampler
from utils.logger import get_logger, SessionLogger, section_header

log = get_logger(__name__)


# ===========================================================================
# SESSION STATE
# ===========================================================================

class Session:
    """
    Per-session state tracker.

    Accumulates round summaries, objection history, Lean progress metrics,
    redirect events, and corpus usage for final reporting.
    """

    def __init__(
        self,
        session_id: str,
        problem_prompt: str,
        framing: str,
        corpus_order: List[str],
    ) -> None:
        self.session_id = session_id
        self.problem_prompt = problem_prompt
        self.framing = framing
        self.corpus_order = corpus_order

        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[str] = None

        self.round_summaries: List[str] = []
        self.past_objections: List[Dict] = []
        self.lean_progress_history: List[int] = []
        self.redirect_events: List[Dict] = []
        self.collected_failures: List[FailureEntry] = []
        self.contributing_analogies: List[str] = []
        self.corpus_files_used: List[str] = []
        self.synthesis_history: List[Dict] = []

        # For termination
        self.last_obstruction_type: str = "unknown"
        self.rounds_without_lean_progress: int = 0
        self.lean_progress_made: bool = False

        # Pending branch specifications from constraint relaxer
        self._pending_branches: List[str] = []
        self._injected_analogy: Optional[str] = None

        # Branch counter for redirect protocol
        self._branch_count: int = 0

    def update(self, synthesis: SynthesisResult) -> None:
        """Update session state from Synthesizer output."""
        self.round_summaries.append(synthesis.round_summary)
        self.synthesis_history.append(synthesis.to_dict())

        # Update Lean progress tracking
        lean_this_round = synthesis.lean_obligations_discharged()
        self.lean_progress_history.append(lean_this_round)
        if lean_this_round > 0:
            self.lean_progress_made = True
            self.rounds_without_lean_progress = 0
        else:
            self.rounds_without_lean_progress += 1

    def add_pending_branches(self, branches: List[str]) -> None:
        self._pending_branches.extend(branches)

    def inject_analogy(self, analogy_text: str) -> None:
        self._injected_analogy = analogy_text

    def pop_analogy(self) -> Optional[str]:
        a = self._injected_analogy
        self._injected_analogy = None
        return a

    def pop_pending_branch(self) -> Optional[str]:
        if self._pending_branches:
            return self._pending_branches.pop(0)
        return None

    def new_branch(self, new_framing: str) -> None:
        self._branch_count += 1
        self.framing = new_framing
        self._pending_branches = []
        log.info(f"Starting branch {self._branch_count} with framing '{new_framing}'")

    def build_context(self) -> str:
        """Build the current context string for the Generator."""
        parts = [self.problem_prompt]
        if self.round_summaries:
            parts.append(
                f"\nPREVIOUS ROUND SUMMARIES:\n"
                + "\n".join(
                    f"  Round {i+1}: {s}"
                    for i, s in enumerate(self.round_summaries[-5:])  # last 5
                )
            )
        if self._pending_branches:
            parts.append(
                f"\nSUGGESTED DIRECTION FROM CONSTRAINT ANALYSIS:\n{self._pending_branches[0]}"
            )
        return "\n".join(parts)

    def save_checkpoint(
        self, round_num: int, scratchpad: Scratchpad, synthesis: SynthesisResult
    ) -> None:
        """Save session checkpoint to disk."""
        checkpoint_dir = Path(config.resolve(config.SESSIONS_DIR))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{self.session_id}.json"

        checkpoint = {
            "session_id": self.session_id,
            "problem_prompt": self.problem_prompt,
            "framing": self.framing,
            "corpus_order": self.corpus_order[:20],
            "start_time": self.start_time,
            "last_round": round_num,
            "round_summaries": self.round_summaries,
            "lean_progress_history": self.lean_progress_history,
            "redirect_events": self.redirect_events,
            "scratchpad": scratchpad.snapshot(),
            "last_synthesis": synthesis.to_dict(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)


# ===========================================================================
# REDIRECT PROTOCOL
# ===========================================================================

def execute_redirect_protocol(
    session: Session,
    scratchpad: Scratchpad,
    signals: TerminationSignals,
    round_num: int,
    constraint_relaxer: ConstraintRelaxerAgent,
    analogy_agent: AnalogyAgent,
    failure_store: FailureStore,
    session_logger: SessionLogger,
) -> Dict[str, Any]:
    """
    Execute the three-layer redirect protocol when 2 of 3 signals fire.

    Returns dict with:
      layer_used: int
      hard_reset: bool
      new_framing: str (if hard reset)
      injection: str (text to inject into Generator)
    """
    session_logger.redirect(round_num, 0, f"Signals: {signals.to_dict()}")

    # Layer 1: Soft redirect — constraint relaxation on most load-bearing assumption
    log.info(f"REDIRECT LAYER 1 (round {round_num})")
    assumption_info = scratchpad.most_load_bearing_assumption()
    assumption_name = assumption_info[0] if assumption_info else "unknown"
    assumption_stmt = assumption_info[1] if assumption_info else ""

    relax_result = constraint_relaxer.generate_variants(scratchpad, focus_assumption=assumption_name)
    injection = relax_result.to_generator_injection()

    if assumption_name:
        n_steps = sum(
            1 for c in scratchpad.conjectures.values()
            if assumption_name in c.get("proof_sketch", "")
        )
        injection = (
            f"The assumption [{assumption_name}] has been relied upon in {n_steps} steps "
            f"without being formally verified. Investigate whether the impossibility result "
            f"survives if this assumption is weakened or removed.\n\n"
        ) + injection

    session_logger.redirect(round_num, 1, f"Focus assumption: {assumption_name}")
    session.redirect_events.append({
        "round": round_num, "layer": 1, "assumption": assumption_name
    })

    # Check if Layer 1 likely had any effect (if no Lean progress in last round anyway)
    if session.rounds_without_lean_progress >= config.ANALOGY_ACTIVATION_ROUND + 1:
        # Layer 2: Medium redirect — analogy injection
        log.info(f"REDIRECT LAYER 2 (round {round_num})")
        top_obligation = scratchpad.top_open_conjecture()
        if top_obligation:
            dead_end_techniques = [de["description"] for de in scratchpad.dead_ends[-3:]]
            analogy = analogy_agent.query(
                current_obligation=top_obligation,
                obstruction_type=session.last_obstruction_type,
                dead_ends=dead_end_techniques,
                round_num=round_num,
            )
            session.inject_analogy(analogy.to_generator_injection())
            if analogy.is_useful:
                session.contributing_analogies.append(analogy.analogous_domain)
                injection += f"\n\n{analogy.to_generator_injection()}"

        session_logger.redirect(round_num, 2, "Analogy agent triggered")
        session.redirect_events.append({"round": round_num, "layer": 2})

    # Check if we need Layer 3 (all three signals fire or persistent stall)
    if signals.all_three_fire or session.rounds_without_lean_progress >= config.LEAN_FLAT_ROUNDS * 2:
        # Layer 3: Hard reset — archive branch, start fresh with new framing
        log.info(f"REDIRECT LAYER 3: hard reset (round {round_num})")

        # Archive current branch to failure memory
        top_conj = scratchpad.get_top_conjecture()
        branch_summary = top_conj["statement"][:200] if top_conj else "(no conjecture)"
        obstruction = session.last_obstruction_type

        failure_entry = FailureEntry(
            session_id=session.session_id,
            round_number=round_num,
            timestamp=datetime.now(timezone.utc).isoformat(),
            obstruction_type=obstruction,
            obstruction_description=f"Branch exhausted after {round_num} rounds. "
                                     f"Last obstruction: {obstruction}",
            branch_summary=branch_summary,
        )
        session.collected_failures.append(failure_entry)

        # Select new framing via Thompson sampling (not the current framing)
        new_framing = failure_store.sample_framing_thompson()
        while new_framing == session.framing and len(config.FRAMING_VARIANTS) > 1:
            new_framing = failure_store.sample_framing_thompson()

        session_logger.redirect(
            round_num, 3,
            f"Hard reset: framing {session.framing} -> {new_framing}"
        )
        session.redirect_events.append({
            "round": round_num, "layer": 3,
            "old_framing": session.framing, "new_framing": new_framing
        })

        return {
            "layer_used": 3,
            "hard_reset": True,
            "new_framing": new_framing,
            "injection": (
                f"The following approach has been fully explored and exhausted:\n"
                f"[{branch_summary}]\n\n"
                f"The specific obstruction encountered was: [{obstruction}].\n"
                f"Do not revisit this direction.\n\n"
                f"Starting fresh with a different angle."
            ),
        }

    return {
        "layer_used": 1 if session.rounds_without_lean_progress < config.ANALOGY_ACTIVATION_ROUND + 1 else 2,
        "hard_reset": False,
        "new_framing": session.framing,
        "injection": injection,
    }


# ===========================================================================
# OUTPUT WRITING
# ===========================================================================

def write_json_output(
    session: Session,
    scored_conjectures: List[ConjectureScore],
    scratchpad: Scratchpad,
    termination_reason: str,
    rounds_completed: int,
) -> Path:
    """Write results.json to outputs/{session_id}/."""
    output_dir = Path(config.resolve(config.OUTPUTS_DIR)) / session.session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "session_id": session.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rounds_completed": rounds_completed,
        "termination_reason": termination_reason,
        "problem_prompt": session.problem_prompt,
        "framing_used": session.framing,
        "corpus_files_used": list(set(session.corpus_files_used)),
        "conjectures": [],
        "failure_memory_entries_added": len(session.collected_failures),
        "redirect_events": session.redirect_events,
        "session_log": session.synthesis_history,
    }

    for rank, score in enumerate(scored_conjectures, 1):
        entry = score.to_dict()
        entry["rank"] = rank
        # Add scratchpad snapshot for this conjecture
        entry["scratchpad_at_termination"] = scratchpad.snapshot()
        entry["deepseek_prover_verdict"] = (
            "valid" if score.deepseek_valid else "invalid"
        )
        results["conjectures"].append(entry)

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"Results written to: {output_path}")
    return output_path


def write_report(
    session: Session,
    scored_conjectures: List[ConjectureScore],
    scratchpad: Scratchpad,
    termination_reason: str,
    rounds_completed: int,
    failure_store: FailureStore,
) -> Path:
    """Write report.md to outputs/{session_id}/."""
    output_dir = Path(config.resolve(config.OUTPUTS_DIR)) / session.session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# NOVA Reasoning System — Session Report",
        f"",
        f"**Session ID:** `{session.session_id}`  ",
        f"**Date:** {session.start_time}  ",
        f"**Rounds completed:** {rounds_completed}  ",
        f"**Termination reason:** {termination_reason}  ",
        f"**Framing used:** {session.framing}  ",
        f"",
        f"---",
        f"",
        f"## Problem Statement",
        f"",
        f"```",
        session.problem_prompt.strip(),
        f"```",
        f"",
        f"---",
        f"",
        f"## Reasoning Trajectory",
        f"",
    ]

    for i, summary in enumerate(session.round_summaries, 1):
        lines.append(f"### Round {i}")
        lines.append(f"")
        lines.append(summary)
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## Top-Ranked Conjecture",
        f"",
    ]

    if scored_conjectures:
        top = scored_conjectures[0]
        lines += [
            f"**Statement:**",
            f"",
            f"> {top.statement}",
            f"",
            f"**Composite Score:** {top.composite_score:.4f}",
            f"",
            f"| Signal | Value |",
            f"|--------|-------|",
            f"| Lean Coverage | {top.lean_coverage:.0%} |",
            f"| Self-Consistency Score | {top.uncertainty_score:.3f} |",
            f"| Structural Surprise | {top.structural_surprise:.3f} |",
            f"| DeepSeek-Prover Agreement | {'Yes' if top.deepseek_valid else 'No'} |",
            f"",
            f"**Lean Obligations:** {top.lean_obligations_discharged} discharged / {top.lean_obligations_total} total",
            f"",
        ]

        if top.lean_proof_fragments:
            lines += [
                f"**Lean Proof Fragment:**",
                f"",
                f"```lean",
                *top.lean_proof_fragments,
                f"```",
                f"",
            ]

        if top.proof_sketch:
            lines += [
                f"**Proof Sketch:**",
                f"",
                top.proof_sketch,
                f"",
            ]
    else:
        lines.append("No conjectures were produced.")
        lines.append("")

    # All conjectures ranked
    if len(scored_conjectures) > 1:
        lines += [
            f"---",
            f"",
            f"## All Conjectures (Ranked)",
            f"",
        ]
        for rank, score in enumerate(scored_conjectures, 1):
            lines.append(
                f"{rank}. **[{score.composite_score:.4f}]** {score.statement[:120]}..."
            )
        lines.append("")

    # Corpus usage
    lines += [
        f"---",
        f"",
        f"## Corpus Usage",
        f"",
        f"Files used in this session:",
        f"",
    ]
    for f in set(session.corpus_files_used):
        lines.append(f"- `{f}`")
    lines.append("")

    # Failure memory
    lines += [
        f"---",
        f"",
        f"## Failure Memory",
        f"",
        f"**Failures added this session:** {len(session.collected_failures)}",
        f"",
    ]
    if session.collected_failures:
        for fe in session.collected_failures[:5]:
            lines.append(
                f"- **Round {fe.round_number}** [{fe.obstruction_type}]: "
                f"{fe.obstruction_description[:100]}..."
            )
    lines.append("")

    # Obstructions
    lines += [
        f"---",
        f"",
        f"## Obstructions and Resolutions",
        f"",
    ]
    for de in scratchpad.dead_ends:
        lines.append(
            f"- **Round {de['round']}** [{de['obstruction_type']}]: "
            f"{de['description']}"
        )
        if de.get("obstruction"):
            lines.append(f"  - Obstruction: {de['obstruction']}")
    lines.append("")

    # Redirect events
    if session.redirect_events:
        lines += [
            f"---",
            f"",
            f"## Redirect Events",
            f"",
        ]
        for re_event in session.redirect_events:
            lines.append(
                f"- Round {re_event['round']} — Layer {re_event['layer']}"
                + (f": assumption `{re_event.get('assumption', '')}`" if re_event.get('assumption') else "")
            )
        lines.append("")

    # Analogies
    if session.contributing_analogies:
        lines += [
            f"---",
            f"",
            f"## Cross-Domain Analogies Used",
            f"",
        ]
        for domain in session.contributing_analogies:
            lines.append(f"- {domain}")
        lines.append("")

    # What remains unproven
    lines += [
        f"---",
        f"",
        f"## What Remains Unproven",
        f"",
    ]
    open_conjectures = [
        (name, entry) for name, entry in scratchpad.conjectures.items()
        if entry.get("lean_coverage", 0.0) < 1.0
    ]
    if open_conjectures:
        for name, entry in open_conjectures:
            coverage = entry.get("lean_coverage", 0.0)
            lines.append(
                f"- **{name}** (Lean coverage: {coverage:.0%}): "
                f"{entry['statement']}"
            )
            if entry.get("lean_result") and entry["lean_result"].get("errors"):
                lines.append(
                    f"  - Lean errors: "
                    + "; ".join(entry["lean_result"]["errors"][:2])
                )
    else:
        lines.append("All conjectures have been either verified or are in established state.")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    log.info(f"Report written to: {report_path}")
    return report_path


# ===========================================================================
# MAIN SESSION RUNNER
# ===========================================================================

def run_session(
    problem_prompt: str,
    max_rounds: int = config.MAX_ROUNDS,
    session_id: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> List[ConjectureScore]:
    """
    Run a complete reasoning session.

    Args:
        problem_prompt: The scientific question (plain text).
        max_rounds:     Maximum number of reasoning rounds.
        session_id:     Session identifier (auto-generated if None).
        resume_from:    Session ID to resume from checkpoint.

    Returns:
        List of scored conjectures, sorted highest to lowest.
    """
    if session_id is None:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    session_logger = SessionLogger(session_id)
    section_header(f"NOVA REASONING SESSION: {session_id}")

    # -----------------------------------------------------------------------
    # 1. Initialize tools and memory
    # -----------------------------------------------------------------------
    log.info("Initializing tools...")
    lean_tool = LeanTool()
    z3_tool = Z3Tool()
    sympy_tool = SymPyTool()
    failure_store = FailureStore()
    corpus_retriever = CorpusRetriever()

    # Build/update corpus index
    corpus_retriever.build_or_update_index()

    # -----------------------------------------------------------------------
    # 2. Initialize agents
    # -----------------------------------------------------------------------
    log.info("Initializing agents...")
    generator = GeneratorAgent(lean_tool, z3_tool, sympy_tool, corpus_retriever)
    adversarial_critic = AdversarialCriticAgent()
    devils_advocate = DevilsAdvocateAgent()
    synthesizer = SynthesizerAgent()
    constraint_relaxer = ConstraintRelaxerAgent()
    analogy_agent_inst = AnalogyAgent(corpus_retriever)
    scorer = ConjectureScorer()

    # -----------------------------------------------------------------------
    # 3. Load failure memory
    # -----------------------------------------------------------------------
    log.info("Loading failure memory...")
    past_failures = failure_store.get_relevant(
        problem_prompt, top_k=config.FAILURE_RETRIEVAL_TOP_K
    )
    failure_context = "\n\n".join(f.to_context_string() for f in past_failures)
    if past_failures:
        log.info(f"Loaded {len(past_failures)} relevant past failures")

    # -----------------------------------------------------------------------
    # 4. Select framing variant via Thompson sampling
    # -----------------------------------------------------------------------
    framing = failure_store.sample_framing_thompson()
    log.info(f"Framing variant selected: {framing}")

    # -----------------------------------------------------------------------
    # 5. DPP-sample corpus ordering
    # -----------------------------------------------------------------------
    log.info("DPP-sampling corpus ordering...")
    dpp_sampler = DPPSampler(failure_store=failure_store)
    try:
        corpus_order = dpp_sampler.sample_paper_subset(
            corpus_retriever, target_papers=10
        )
    except Exception as e:
        log.warning(f"DPP sampling failed: {e}. Using default ordering.")
        corpus_order = corpus_retriever.get_paper_names()[:10]

    # -----------------------------------------------------------------------
    # 6. Initialize session
    # -----------------------------------------------------------------------
    scratchpad = Scratchpad()
    session = Session(session_id, problem_prompt, framing, corpus_order)

    # Resume from checkpoint if requested
    start_round = 1
    if resume_from:
        start_round, scratchpad, session = _load_checkpoint(resume_from, session)
        log.info(f"Resuming from session {resume_from} at round {start_round}")

    # -----------------------------------------------------------------------
    # 7. Initialize DeepSeek-Prover
    # -----------------------------------------------------------------------
    from utils.model_loader import get_deepseek_prover
    deepseek_prover = get_deepseek_prover()

    # -----------------------------------------------------------------------
    # 8. Main round loop
    # -----------------------------------------------------------------------
    termination_reason = "max_rounds_reached"

    for round_num in range(start_round, max_rounds + 1):
        scratchpad.advance_round()
        section_header(f"ROUND {round_num} / {max_rounds}")
        session_logger.round_start(round_num, framing=session.framing)

        # Track Lean progress for this round
        lean_obligations_before = lean_tool.new_obligations_this_session()

        # -------------------------------------------------------------------
        # Generator runs free for GENERATOR_FREE_STEPS
        # -------------------------------------------------------------------
        analogy_injection = session.pop_analogy()
        pending_branch = session.pop_pending_branch()
        context = session.build_context()
        if pending_branch:
            context += f"\n\nSUGGESTED EXPLORATION DIRECTION:\n{pending_branch}"

        generator_output: GeneratorOutput = generator.run(
            context=context,
            scratchpad=scratchpad,
            framing=session.framing,
            injected_failures=failure_context if round_num == 1 else None,
            injected_analogy=analogy_injection,
            free_steps=config.GENERATOR_FREE_STEPS,
            round_num=round_num,
        )

        # Track corpus files used
        for q in generator_output.corpus_queries:
            # corpus queries don't have file names directly, but we can note them
            pass

        # -------------------------------------------------------------------
        # Lean verification of all new claims
        # -------------------------------------------------------------------
        lean_results = lean_tool.verify_all_new_claims(scratchpad)

        lean_obligations_after = lean_tool.new_obligations_this_session()
        lean_this_round = lean_obligations_after - lean_obligations_before

        # -------------------------------------------------------------------
        # Critics run (in parallel via threads)
        # -------------------------------------------------------------------
        adversarial_result_holder: List[AdversarialResult] = [None]
        devils_result_holder = [None]

        def run_adversarial():
            top_conj = scratchpad.get_top_conjecture()
            proof_sketch = top_conj.get("proof_sketch", "") if top_conj else generator_output.final_statement
            adversarial_result_holder[0] = adversarial_critic.critique(
                proof_sketch=proof_sketch,
                scratchpad_summary=scratchpad.context_summary(),
                previous_objections=session.past_objections,
                round_num=round_num,
            )

        def run_devils():
            top_conj = scratchpad.get_top_conjecture()
            hypothesis = top_conj.get("statement", "") if top_conj else generator_output.final_statement
            proof_sketch = top_conj.get("proof_sketch", "") if top_conj else ""
            # Retrieve corpus context for Devil's Advocate
            corpus_chunks = corpus_retriever.retrieve(hypothesis[:200], top_k=3)
            corpus_ctx = "\n\n".join(c.text[:200] for c in corpus_chunks) if corpus_chunks else None
            devils_result_holder[0] = devils_advocate.critique(
                current_hypothesis=hypothesis,
                proof_sketch=proof_sketch,
                scratchpad_summary=scratchpad.context_summary(),
                corpus_context=corpus_ctx,
                round_num=round_num,
            )

        adv_thread = threading.Thread(target=run_adversarial)
        dev_thread = threading.Thread(target=run_devils)
        adv_thread.start()
        dev_thread.start()
        adv_thread.join(timeout=120)
        dev_thread.join(timeout=120)

        adversarial_result = adversarial_result_holder[0]
        devils_result = devils_result_holder[0]

        # Fallback if threads timed out
        if adversarial_result is None:
            log.warning("Adversarial critic timed out — using stub result")
            from agents.adversarial_critic import AdversarialResult as AR
            adversarial_result = AR(
                objection_type="unknown", assumption_challenged="timeout",
                proof_step_targeted=0, formal_objection="Critic timed out",
                counterexample_sketch=None, what_would_fix_it="", is_exhausted=True
            )
        if devils_result is None:
            log.warning("Devil's advocate timed out — using stub result")
            from agents.devils_advocate import DevilsAdvocateResult as DR
            devils_result = DR(
                closest_known_result="timeout", novelty_challenge="timeout",
                motivation_challenge="", what_is_actually_new="", verdict="MARGINALLY_NOVEL"
            )

        # Store objection for novelty tracking
        session.past_objections.append(adversarial_result.to_dict())
        session.last_obstruction_type = adversarial_result.objection_type

        # Log critic results
        session_logger.critic_objection(
            round_num=round_num,
            critic="adversarial",
            objection_type=adversarial_result.objection_type,
            assumption_challenged=adversarial_result.assumption_challenged,
            proof_step_targeted=adversarial_result.proof_step_targeted,
            tag="DEGENERATE" if adversarial_result.is_exhausted else "OPEN",
        )

        # -------------------------------------------------------------------
        # DeepSeek-Prover evaluates top conjecture
        # -------------------------------------------------------------------
        top_conj = scratchpad.get_top_conjecture()
        if top_conj:
            deepseek_result = deepseek_prover.evaluate(
                top_conj.get("proof_sketch", "")
            )
        else:
            deepseek_result = {"valid": False, "confidence": 0.0, "issues": []}

        # -------------------------------------------------------------------
        # Synthesizer processes everything
        # -------------------------------------------------------------------
        synthesis: SynthesisResult = synthesizer.process(
            generator_output=generator_output,
            lean_results=lean_results,
            adversarial_result=adversarial_result,
            devils_result=devils_result,
            deepseek_result=deepseek_result,
            scratchpad=scratchpad,
            round_num=round_num,
            session=session,
        )

        session.update(synthesis)

        # Log Lean results
        for lr in lean_results:
            session_logger.lean_result(
                round_num=round_num,
                claim_name="",
                success=lr.success,
                obligations_discharged=lr.obligations_discharged,
                obligations_remaining=lr.obligations_remaining,
                errors=lr.errors,
                semantic_drift=lr.semantic_drift,
            )

        # Store adversarial failures in failure list
        if not adversarial_result.is_exhausted:
            failure_entry = FailureEntry(
                session_id=session_id,
                round_number=round_num,
                timestamp=datetime.now(timezone.utc).isoformat(),
                obstruction_type=adversarial_result.objection_type,
                obstruction_description=adversarial_result.formal_objection,
                assumption_challenged=adversarial_result.assumption_challenged,
                proof_step_failed=adversarial_result.proof_step_targeted,
                branch_summary=generator_output.final_statement[:200],
            )
            session.collected_failures.append(failure_entry)

        # -------------------------------------------------------------------
        # Check termination signals
        # -------------------------------------------------------------------
        signals: TerminationSignals = synthesis.termination_signals or TerminationSignals()

        if signals.all_three_fire:
            log.info("All 3 termination signals fired. Ending session.")
            termination_reason = "all_signals_fired"
            session_logger.termination(termination_reason, round_num)
            session.save_checkpoint(round_num, scratchpad, synthesis)
            break

        if signals.two_fire:
            log.info(f"2/3 termination signals fired at round {round_num}. Executing redirect.")
            redirect = execute_redirect_protocol(
                session=session,
                scratchpad=scratchpad,
                signals=signals,
                round_num=round_num,
                constraint_relaxer=constraint_relaxer,
                analogy_agent=analogy_agent_inst,
                failure_store=failure_store,
                session_logger=session_logger,
            )
            if redirect["hard_reset"]:
                scratchpad = Scratchpad()
                session.new_branch(redirect["new_framing"])
                # Inject the archived branch context into next round
                session._pending_branches.insert(0, redirect["injection"])

        # -------------------------------------------------------------------
        # Every CONSTRAINT_RELAXATION_INTERVAL rounds: constraint relaxation
        # -------------------------------------------------------------------
        if round_num % config.CONSTRAINT_RELAXATION_INTERVAL == 0:
            relax_result = constraint_relaxer.generate_variants(scratchpad)
            if relax_result.new_branch_specification:
                session.add_pending_branches([relax_result.new_branch_specification])

        # -------------------------------------------------------------------
        # Stall detection: trigger analogy agent
        # -------------------------------------------------------------------
        if session.rounds_without_lean_progress >= config.ANALOGY_ACTIVATION_ROUND:
            top_ob = scratchpad.top_open_conjecture()
            if top_ob:
                dead_techniques = [de["description"] for de in scratchpad.dead_ends[-3:]]
                analogy = analogy_agent_inst.query(
                    current_obligation=top_ob,
                    obstruction_type=session.last_obstruction_type,
                    dead_ends=dead_techniques,
                    round_num=round_num,
                )
                if analogy.is_useful:
                    session.inject_analogy(analogy.to_generator_injection())
                    session.contributing_analogies.append(analogy.analogous_domain)

        # Round summary
        session_logger.round_end(
            round_num,
            summary=synthesis.round_summary[:200],
            lean_discharged=lean_this_round,
            termination_signals=signals.to_dict(),
        )

        # Save checkpoint every round
        session.save_checkpoint(round_num, scratchpad, synthesis)

    else:
        # Loop completed without break — max rounds reached
        termination_reason = "max_rounds_reached"
        session_logger.termination(termination_reason, max_rounds)

    # -----------------------------------------------------------------------
    # 9. Score surviving conjectures
    # -----------------------------------------------------------------------
    section_header("SCORING CONJECTURES")
    scored = scorer.score_all(scratchpad.conjectures, deepseek_prover, corpus_retriever)

    # -----------------------------------------------------------------------
    # 10. Update failure memory
    # -----------------------------------------------------------------------
    failure_store.add_session_failures(session.collected_failures)
    failure_store.decay_salience()
    failure_store.update_framing_stats(session.framing, session.lean_progress_made)

    # Update paper quality weights based on "contradicts_corpus" failures
    for fe in session.collected_failures:
        if fe.obstruction_type == "contradicts_corpus" and fe.branch_summary:
            # Try to identify which paper was retrieved — use the branch summary as a proxy
            for fname in corpus_retriever.get_paper_names():
                if fname.lower() in fe.branch_summary.lower():
                    failure_store.update_paper_quality(fname, delta=-0.1)

    failure_store.save_session(
        session_id=session_id,
        start_time=session.start_time,
        end_time=datetime.now(timezone.utc).isoformat(),
        total_rounds=len(session.round_summaries),
        termination_reason=termination_reason,
        best_conjecture=scored[0].statement if scored else None,
        best_lean_coverage=scored[0].lean_coverage if scored else 0.0,
        framing_used=session.framing,
        corpus_subset=session.corpus_order[:10],
    )

    # -----------------------------------------------------------------------
    # 11. Write outputs
    # -----------------------------------------------------------------------
    section_header("WRITING OUTPUTS")
    rounds_done = len(session.round_summaries)
    write_json_output(session, scored, scratchpad, termination_reason, rounds_done)
    write_report(session, scored, scratchpad, termination_reason, rounds_done, failure_store)

    session_logger.print_summary()
    print(f"\nSession complete. Results written to: {config.resolve(config.OUTPUTS_DIR)}/{session_id}/")

    return scored


# ===========================================================================
# CHECKPOINT LOADING
# ===========================================================================

def _load_checkpoint(session_id: str, current_session: Session):
    """Load a session checkpoint and return (start_round, scratchpad, session)."""
    checkpoint_path = (
        Path(config.resolve(config.SESSIONS_DIR)) / f"{session_id}.json"
    )
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return 1, Scratchpad(), current_session

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    scratchpad = Scratchpad.from_dict(checkpoint.get("scratchpad", {}))
    current_session.round_summaries = checkpoint.get("round_summaries", [])
    current_session.lean_progress_history = checkpoint.get("lean_progress_history", [])
    current_session.redirect_events = checkpoint.get("redirect_events", [])
    start_round = checkpoint.get("last_round", 1) + 1

    log.info(f"Loaded checkpoint: resuming from round {start_round}")
    return start_round, scratchpad, current_session


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOVA Reasoning System — Scientific hypothesis discovery via multi-agent reasoning"
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="Path to problem prompt text file",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=config.MAX_ROUNDS,
        help=f"Maximum rounds (default: {config.MAX_ROUNDS})",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID (auto-generated if not specified)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a previous session ID",
    )

    args = parser.parse_args()

    # Read problem prompt
    problem_path = Path(args.problem)
    if not problem_path.exists():
        print(f"Error: problem prompt file not found: {problem_path}")
        sys.exit(1)

    with open(problem_path) as f:
        problem_prompt = f.read().strip()

    if not problem_prompt:
        print("Error: problem prompt file is empty.")
        sys.exit(1)

    log.info(f"Problem prompt loaded from: {problem_path}")
    log.info(f"Problem length: {len(problem_prompt)} characters")

    # Run session
    scored = run_session(
        problem_prompt=problem_prompt,
        max_rounds=args.rounds,
        session_id=args.session_id,
        resume_from=args.resume,
    )

    if scored:
        print(f"\nTop conjecture (score={scored[0].composite_score:.4f}):")
        print(f"  {scored[0].statement}")
        print(f"  Lean coverage: {scored[0].lean_coverage:.0%}")
        print(f"  DeepSeek agreement: {scored[0].deepseek_valid}")
    else:
        print("\nNo conjectures produced in this session.")


if __name__ == "__main__":
    main()
