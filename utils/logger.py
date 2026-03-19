"""
utils/logger.py — Structured session logging for the NOVA Reasoning System.

Provides two logging surfaces:
  1. A standard Python logger with Rich console formatting for human-readable
     terminal output during a session.
  2. A structured event log (list of dicts) accumulated in memory during a
     session and serialized to the session JSON and report at termination.

Usage:
    from utils.logger import get_logger, SessionLogger

    log = get_logger(__name__)
    log.info("Starting round %d", round_num)

    session_log = SessionLogger(session_id)
    session_log.event("round_start", round=1, framing="assumption")
    events = session_log.get_events()
"""

import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

import config

# ---------------------------------------------------------------------------
# Module-level console (shared across all loggers in the process)
# ---------------------------------------------------------------------------

if _RICH_AVAILABLE:
    _console = Console(
        theme=Theme({
            "logging.level.debug":   "dim cyan",
            "logging.level.info":    "bold green",
            "logging.level.warning": "bold yellow",
            "logging.level.error":   "bold red",
            "logging.level.critical":"bold white on red",
        }),
        stderr=True,
    )
else:
    _console = None

_root_configured = False


def _configure_root() -> None:
    """Configure the root logger once (idempotent)."""
    global _root_configured
    if _root_configured:
        return
    _root_configured = True

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    if _RICH_AVAILABLE:
        handler = RichHandler(
            console=_console,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "urllib3", "requests", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Structured session event log
# ---------------------------------------------------------------------------

class SessionLogger:
    """
    Accumulates structured events during a session.

    Each event is a dict with at minimum:
        timestamp (ISO 8601), event_type (str), session_id (str)
    plus any keyword arguments passed by the caller.

    The log is kept in memory and serialized to the session JSON at termination.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._events: List[Dict[str, Any]] = []
        self._log = get_logger(f"session.{session_id}")

    def event(self, event_type: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Record a structured event.

        Args:
            event_type: A short snake_case identifier for the event class,
                        e.g. "round_start", "lean_success", "critic_objection".
            **kwargs:   Any additional structured fields for this event.

        Returns:
            The event dict that was recorded.
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "session_id": self.session_id,
        }
        entry.update(kwargs)
        self._events.append(entry)

        # Mirror to the Python logger for terminal output
        log_msg = f"[{event_type}] " + " ".join(
            f"{k}={v!r}" for k, v in kwargs.items()
            if k not in ("content", "proof_sketch", "lean_output")  # suppress bulky fields
        )
        self._log.debug(log_msg)
        return entry

    def round_start(self, round_num: int, **kwargs: Any) -> None:
        self._log.info(f"=== ROUND {round_num} START ===")
        self.event("round_start", round=round_num, **kwargs)

    def round_end(self, round_num: int, summary: str, **kwargs: Any) -> None:
        self._log.info(f"=== ROUND {round_num} END: {summary} ===")
        self.event("round_end", round=round_num, summary=summary, **kwargs)

    def lean_result(
        self,
        round_num: int,
        claim_name: str,
        success: bool,
        obligations_discharged: int,
        obligations_remaining: int,
        errors: Optional[List[str]] = None,
        semantic_drift: bool = False,
    ) -> None:
        status = "SUCCESS" if success else "FAIL"
        self._log.info(
            f"Lean [{status}] {claim_name}: "
            f"{obligations_discharged} discharged, "
            f"{obligations_remaining} remaining"
        )
        self.event(
            "lean_result",
            round=round_num,
            claim_name=claim_name,
            success=success,
            obligations_discharged=obligations_discharged,
            obligations_remaining=obligations_remaining,
            errors=errors or [],
            semantic_drift=semantic_drift,
        )

    def critic_objection(
        self,
        round_num: int,
        critic: str,
        objection_type: str,
        assumption_challenged: str,
        proof_step_targeted: int,
        tag: str = "OPEN",
    ) -> None:
        self._log.info(
            f"[{critic}] Round {round_num}: {objection_type} @ step {proof_step_targeted}"
            f" — assumption: {assumption_challenged!r} [{tag}]"
        )
        self.event(
            "critic_objection",
            round=round_num,
            critic=critic,
            objection_type=objection_type,
            assumption_challenged=assumption_challenged,
            proof_step_targeted=proof_step_targeted,
            tag=tag,
        )

    def redirect(self, round_num: int, layer: int, reason: str) -> None:
        self._log.warning(
            f"REDIRECT LAYER {layer} triggered at round {round_num}: {reason}"
        )
        self.event("redirect", round=round_num, layer=layer, reason=reason)

    def termination(self, reason: str, rounds_completed: int) -> None:
        self._log.info(
            f"SESSION TERMINATED after {rounds_completed} rounds. Reason: {reason}"
        )
        self.event("termination", reason=reason, rounds_completed=rounds_completed)

    def get_events(self) -> List[Dict[str, Any]]:
        """Return the full structured event list."""
        return list(self._events)

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Return only events of a specific type."""
        return [e for e in self._events if e["event_type"] == event_type]

    def print_summary(self) -> None:
        """Print a brief summary of session statistics to the terminal."""
        total = len(self._events)
        by_type: Dict[str, int] = {}
        for e in self._events:
            by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1

        self._log.info(f"Session event summary ({total} total events):")
        for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            self._log.info(f"  {etype}: {count}")


# ---------------------------------------------------------------------------
# Progress utilities (wraps tqdm if available, otherwise plain print)
# ---------------------------------------------------------------------------

def progress_bar(iterable, desc: str = "", total: Optional[int] = None):
    """Wrap an iterable in a progress bar if tqdm is available."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, file=sys.stderr)
    except ImportError:
        return iterable


def section_header(title: str, width: int = 70) -> None:
    """Print a visible section header to stderr."""
    bar = "=" * width
    print(f"\n{bar}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{bar}\n", file=sys.stderr)
