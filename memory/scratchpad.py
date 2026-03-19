"""
memory/scratchpad.py — In-session structured working memory.

The scratchpad is the Generator's working memory for a session. It maintains
four typed namespaces:
  axioms      — statements taken as given, not questioned
  established — claims that have passed Lean verification
  assumptions — claims that are being relied upon but not yet proven
  conjectures — current proof targets
  dead_ends   — approaches that have been exhausted with a specific obstruction

Key invariant: the Generator may not reference a result without first writing
it to the scratchpad. The system checks this by scanning Generator output for
names that appear in previous established/assumption entries.

The scratchpad is serialized to the session checkpoint JSON at the end of
every round so that the session can be resumed after interruption.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


class Scratchpad:
    """
    In-session structured working memory.

    All writes go through typed methods that enforce invariants.
    The underlying dict is directly serializable to JSON.
    """

    VALID_TYPES = frozenset({"axiom", "established", "assumption", "conjecture"})

    def __init__(self, initial_state: Optional[Dict] = None) -> None:
        if initial_state:
            self.axioms: Dict[str, Any] = initial_state.get("axioms", {})
            self.established: Dict[str, Any] = initial_state.get("established", {})
            self.assumptions: Dict[str, Any] = initial_state.get("assumptions", {})
            self.conjectures: Dict[str, Any] = initial_state.get("conjectures", {})
            self.dead_ends: List[Dict] = initial_state.get("dead_ends", [])
        else:
            self.axioms: Dict[str, Any] = {}
            self.established: Dict[str, Any] = {}
            self.assumptions: Dict[str, Any] = {}
            self.conjectures: Dict[str, Any] = {}
            self.dead_ends: List[Dict] = []

        self._round_counter: int = 0

    # -----------------------------------------------------------------------
    # PUBLIC: WRITE OPERATIONS
    # -----------------------------------------------------------------------

    def write(self, name: str, statement: str, entry_type: str, **kwargs) -> None:
        """
        Write a named entry to the scratchpad.

        Args:
            name:       Unique identifier for this result.
            statement:  The mathematical statement.
            entry_type: "axiom" | "established" | "assumption" | "conjecture"
            **kwargs:   Additional metadata (proof_sketch, lean_proof, etc.)
        """
        if entry_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid entry type: {entry_type!r}. "
                f"Must be one of: {sorted(self.VALID_TYPES)}"
            )

        entry = {
            "statement": statement,
            "round": self._round_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry.update(kwargs)

        if entry_type == "axiom":
            self.axioms[name] = entry
            log.debug(f"Scratchpad AXIOM: {name}")

        elif entry_type == "established":
            entry["lean_proof"] = kwargs.get("lean_proof", "")
            self.established[name] = entry
            log.debug(f"Scratchpad ESTABLISHED: {name}")

        elif entry_type == "assumption":
            entry["challenged"] = False
            self.assumptions[name] = entry
            log.debug(f"Scratchpad ASSUMPTION: {name}")

        elif entry_type == "conjecture":
            entry["lean_coverage"] = kwargs.get("lean_coverage", 0.0)
            entry["proof_sketch"] = kwargs.get("proof_sketch", "")
            entry["lean_result"] = kwargs.get("lean_result", None)
            self.conjectures[name] = entry
            log.debug(f"Scratchpad CONJECTURE: {name}")

    def add_axiom(self, name: str, statement: str) -> None:
        self.write(name, statement, "axiom")

    def add_established(self, name: str, statement: str, lean_proof: str = "", **kwargs) -> None:
        self.write(name, statement, "established", lean_proof=lean_proof, **kwargs)

    def add_assumption(self, name: str, statement: str, **kwargs) -> None:
        self.write(name, statement, "assumption", **kwargs)

    def add_conjecture(
        self,
        name: str,
        statement: str,
        proof_sketch: str = "",
        lean_coverage: float = 0.0,
        **kwargs,
    ) -> None:
        self.write(
            name, statement, "conjecture",
            proof_sketch=proof_sketch,
            lean_coverage=lean_coverage,
            **kwargs,
        )

    def add_dead_end(
        self,
        description: str,
        obstruction: str,
        obstruction_type: str = "unknown",
        branch_summary: Optional[str] = None,
    ) -> None:
        self.dead_ends.append({
            "description": description,
            "obstruction": obstruction,
            "obstruction_type": obstruction_type,
            "branch_summary": branch_summary or description,
            "round": self._round_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        log.debug(f"Scratchpad DEAD_END: {description[:60]}...")

    # -----------------------------------------------------------------------
    # PUBLIC: PROMOTION
    # -----------------------------------------------------------------------

    def promote_to_established(self, name: str, lean_proof: str) -> bool:
        """
        Promote an assumption or conjecture to established status after Lean verification.
        Returns True if promotion occurred.
        """
        if name in self.assumptions:
            entry = self.assumptions.pop(name)
            entry["lean_proof"] = lean_proof
            entry["promoted_at_round"] = self._round_counter
            self.established[name] = entry
            log.info(f"PROMOTED to established: {name}")
            return True

        if name in self.conjectures:
            entry = self.conjectures.pop(name)
            entry["lean_proof"] = lean_proof
            entry["promoted_at_round"] = self._round_counter
            entry.pop("lean_coverage", None)
            self.established[name] = entry
            log.info(f"PROMOTED conjecture to established: {name}")
            return True

        return False

    def mark_assumption_challenged(self, name: str) -> None:
        """Flag an assumption as having been challenged by a critic."""
        if name in self.assumptions:
            self.assumptions[name]["challenged"] = True

    def update_lean_coverage(self, name: str, coverage: float, lean_result: Any = None) -> None:
        """Update the Lean coverage for a conjecture."""
        if name in self.conjectures:
            self.conjectures[name]["lean_coverage"] = coverage
            if lean_result is not None:
                self.conjectures[name]["lean_result"] = lean_result

    # -----------------------------------------------------------------------
    # PUBLIC: READ OPERATIONS
    # -----------------------------------------------------------------------

    def read(self, name: str) -> Optional[Dict[str, Any]]:
        """Read any entry by name, regardless of type."""
        for namespace in (self.axioms, self.established, self.assumptions, self.conjectures):
            if name in namespace:
                return namespace[name]
        return None

    def list_all(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Return a summary of all entries, organized by type.
        Format: {"axioms": [(name, statement_prefix), ...], ...}
        """
        def _summarize(ns: Dict) -> List[Tuple[str, str]]:
            return [(k, v["statement"][:100]) for k, v in ns.items()]

        return {
            "axioms":      _summarize(self.axioms),
            "established": _summarize(self.established),
            "assumptions": _summarize(self.assumptions),
            "conjectures": _summarize(self.conjectures),
            "dead_ends":   [(d["description"][:80],) for d in self.dead_ends],
        }

    def get_top_conjecture(self) -> Optional[Dict[str, Any]]:
        """
        Return the conjecture with the highest Lean coverage.
        If tied, return the most recently written one.
        """
        if not self.conjectures:
            return None
        return max(
            self.conjectures.values(),
            key=lambda c: (c.get("lean_coverage", 0.0), c.get("round", 0)),
        )

    def top_open_conjecture(self) -> Optional[str]:
        """
        Return the statement of the highest-coverage conjecture that is NOT
        yet fully established (lean_coverage < 1.0).
        """
        candidates = [
            (v.get("lean_coverage", 0.0), k, v)
            for k, v in self.conjectures.items()
            if v.get("lean_coverage", 0.0) < 1.0
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][2]["statement"]

    def unchallenged_assumptions(self) -> List[Tuple[str, str]]:
        """Return (name, statement) pairs for assumptions not yet challenged."""
        return [
            (name, entry["statement"])
            for name, entry in self.assumptions.items()
            if not entry.get("challenged", False)
        ]

    def most_load_bearing_assumption(self) -> Optional[Tuple[str, str]]:
        """
        Return the assumption that appears most often in conjecture proof sketches
        (the one doing the most load-bearing work).
        """
        if not self.assumptions:
            return None

        counts: Dict[str, int] = {name: 0 for name in self.assumptions}
        for conj in self.conjectures.values():
            sketch = conj.get("proof_sketch", "")
            for name in self.assumptions:
                if name in sketch:
                    counts[name] += 1

        most_cited = max(counts.items(), key=lambda x: x[1])
        if most_cited[1] == 0:
            # No assumption is explicitly cited — return first unchallenged
            for name, entry in self.assumptions.items():
                if not entry.get("challenged", False):
                    return name, entry["statement"]
        return most_cited[0], self.assumptions[most_cited[0]]["statement"]

    def is_dead_end(
        self,
        description: str,
        similarity_threshold: float = 0.85,
    ) -> Optional[Dict]:
        """
        Check if a description is similar to any known dead end.
        Uses embedding similarity. Returns the dead end entry if similar enough.
        """
        if not self.dead_ends:
            return None

        try:
            from utils.model_loader import embed, cosine_similarity
            desc_emb = embed([description])[0]
            dead_embs = embed([d["description"] for d in self.dead_ends])

            for dead_end, dead_emb in zip(self.dead_ends, dead_embs):
                sim = cosine_similarity(desc_emb, dead_emb)
                if sim >= similarity_threshold:
                    log.info(
                        f"Dead-end similarity {sim:.3f} detected for: {description[:60]}..."
                    )
                    return dead_end
        except Exception as e:
            log.warning(f"Dead-end similarity check failed: {e}")

        return None

    # -----------------------------------------------------------------------
    # PUBLIC: SERIALIZATION
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire scratchpad to a JSON-compatible dict."""
        return {
            "axioms":      self.axioms,
            "established": self.established,
            "assumptions": self.assumptions,
            "conjectures": self.conjectures,
            "dead_ends":   self.dead_ends,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Scratchpad":
        """Reconstruct a scratchpad from a serialized dict."""
        return cls(initial_state=data)

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy for checkpoint storage."""
        return deepcopy(self.to_dict())

    def context_summary(self) -> str:
        """
        Generate a compact text summary of current scratchpad state,
        suitable for injection into Generator/critic prompts.
        """
        lines = ["=== SCRATCHPAD STATE ==="]

        if self.axioms:
            lines.append("\nAXIOMS (taken as given):")
            for name, entry in self.axioms.items():
                lines.append(f"  [{name}]: {entry['statement']}")

        if self.established:
            lines.append("\nESTABLISHED (Lean-verified):")
            for name, entry in self.established.items():
                lines.append(f"  [{name}]: {entry['statement']}")

        if self.assumptions:
            lines.append("\nASSUMPTIONS (unverified, relied upon):")
            for name, entry in self.assumptions.items():
                challenged_flag = " [CHALLENGED]" if entry.get("challenged") else ""
                lines.append(f"  [{name}]{challenged_flag}: {entry['statement']}")

        if self.conjectures:
            lines.append("\nCONJECTURES (current targets):")
            for name, entry in self.conjectures.items():
                coverage = entry.get("lean_coverage", 0.0)
                lines.append(
                    f"  [{name}] (Lean coverage={coverage:.0%}): {entry['statement']}"
                )

        if self.dead_ends:
            lines.append("\nDEAD ENDS (do not revisit):")
            for de in self.dead_ends[-5:]:  # only most recent 5
                lines.append(
                    f"  [Round {de['round']}] {de['obstruction_type']}: "
                    f"{de['description'][:80]}..."
                )

        lines.append("\n=== END SCRATCHPAD ===")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # PUBLIC: ROUND MANAGEMENT
    # -----------------------------------------------------------------------

    def advance_round(self) -> None:
        """Called at the start of each round to update the round counter."""
        self._round_counter += 1

    @property
    def current_round(self) -> int:
        return self._round_counter

    def get_conjectures_needing_verification(self) -> List[Tuple[str, Dict]]:
        """Return (name, entry) for conjectures that have not been Lean-verified."""
        return [
            (name, entry)
            for name, entry in self.conjectures.items()
            if entry.get("lean_coverage", 0.0) == 0.0
               and entry.get("lean_result") is None
        ]
