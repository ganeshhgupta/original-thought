"""
memory/failure_store.py — Persistent SQLite-backed failure memory.

Every failed proof attempt is stored here with a structured taxonomy of WHY
it failed. At the start of each new session, the most relevant failures are
retrieved and injected into the Generator's initial context.

This replicates the human mechanism of carrying forward failed approaches
as updated intuition. The system does not re-attempt approaches that are
similar to stored failures unless the obstruction has been specifically addressed.

Schema: see config documentation.

Key design principles:
  1. Salience decays multiplicatively after each session (FAILURE_SALIENCE_DECAY).
     Old failures fade unless they keep being relevant.
  2. Relevance is computed by embedding similarity between the current problem
     context and the stored failure description.
  3. Quality weights for DPP sampling are stored here and updated only when
     a paper retrieval correlates with a "contradicts_corpus" failure.
  4. Thompson sampling parameters for framing variants are stored and updated
     based on whether Lean obligations were discharged.
"""

import json
import os
import pickle
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import config
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    assumption_challenged TEXT,
    technique_attempted TEXT,
    obstruction_type TEXT CHECK(obstruction_type IN (
        'logical_gap', 'missing_lemma', 'contradicts_corpus',
        'assumption_violation', 'algebraic_obstruction', 'dimensional_mismatch',
        'complexity_barrier', 'unknown'
    )),
    obstruction_description TEXT NOT NULL,
    proof_step_failed INTEGER,
    branch_summary TEXT,
    salience REAL DEFAULT 1.0,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time TEXT,
    end_time TEXT,
    total_rounds INTEGER,
    termination_reason TEXT,
    best_conjecture TEXT,
    best_lean_coverage REAL,
    framing_used TEXT,
    corpus_subset TEXT
);

CREATE TABLE IF NOT EXISTS framing_stats (
    framing TEXT PRIMARY KEY,
    alpha REAL DEFAULT 1.0,
    beta REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS paper_quality_weights (
    source_file TEXT PRIMARY KEY,
    quality REAL DEFAULT 1.0,
    last_updated TEXT
);

CREATE INDEX IF NOT EXISTS idx_failures_session ON failures(session_id);
CREATE INDEX IF NOT EXISTS idx_failures_obstruction ON failures(obstruction_type);
CREATE INDEX IF NOT EXISTS idx_failures_salience ON failures(salience DESC);
"""


@dataclass
class FailureEntry:
    """A single failure record."""
    session_id: str
    round_number: int
    timestamp: str
    obstruction_description: str
    obstruction_type: str = "unknown"
    assumption_challenged: Optional[str] = None
    technique_attempted: Optional[str] = None
    proof_step_failed: Optional[int] = None
    branch_summary: Optional[str] = None
    salience: float = 1.0
    embedding: Optional[List[float]] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)  # don't serialize raw embeddings to JSON
        return d

    def to_context_string(self) -> str:
        """Format as a concise string for injection into Generator context."""
        lines = [
            f"[Past failure — session {self.session_id}, round {self.round_number}]",
            f"  Obstruction: {self.obstruction_type}",
            f"  Description: {self.obstruction_description}",
        ]
        if self.assumption_challenged:
            lines.append(f"  Assumption challenged: {self.assumption_challenged}")
        if self.technique_attempted:
            lines.append(f"  Technique attempted: {self.technique_attempted}")
        if self.branch_summary:
            lines.append(f"  Branch summary: {self.branch_summary}")
        return "\n".join(lines)


class FailureStore:
    """
    SQLite-backed persistent storage for proof failures and session metadata.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = config.resolve(db_path or config.FAILURE_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    # -----------------------------------------------------------------------
    # PUBLIC: FAILURE CRUD
    # -----------------------------------------------------------------------

    def add_failure(self, entry: FailureEntry) -> int:
        """
        Insert a new failure record. Computes and stores the embedding.
        Returns the new row ID.
        """
        # Compute embedding for relevance retrieval
        if entry.embedding is None:
            try:
                from utils.model_loader import embed_single
                embed_text = f"{entry.obstruction_description} {entry.assumption_challenged or ''} {entry.technique_attempted or ''}"
                entry.embedding = embed_single(embed_text)
            except Exception as e:
                log.warning(f"Could not embed failure entry: {e}")
                entry.embedding = None

        emb_blob = pickle.dumps(entry.embedding) if entry.embedding is not None else None

        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO failures (
                    session_id, round_number, timestamp,
                    assumption_challenged, technique_attempted,
                    obstruction_type, obstruction_description,
                    proof_step_failed, branch_summary, salience, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.session_id,
                    entry.round_number,
                    entry.timestamp,
                    entry.assumption_challenged,
                    entry.technique_attempted,
                    entry.obstruction_type,
                    entry.obstruction_description,
                    entry.proof_step_failed,
                    entry.branch_summary,
                    entry.salience,
                    emb_blob,
                ),
            )
            return cursor.lastrowid

    def add_session_failures(self, failures: List[FailureEntry]) -> None:
        """Batch-add failures from a completed session."""
        for f in failures:
            try:
                self.add_failure(f)
            except Exception as e:
                log.error(f"Failed to store failure entry: {e}")

    def get_relevant(
        self,
        context: str,
        top_k: int = config.FAILURE_RETRIEVAL_TOP_K,
        min_salience: float = config.FAILURE_SALIENCE_MIN,
    ) -> List[FailureEntry]:
        """
        Retrieve the most relevant past failures for the current context.

        Relevance = salience × cosine_similarity(context_embedding, failure_embedding).
        Falls back to recency-ordered if embeddings are not available.
        """
        # Get all failures above salience threshold
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM failures WHERE salience >= ? ORDER BY salience DESC",
                (min_salience,),
            ).fetchall()

        if not rows:
            return []

        entries = [self._row_to_entry(row) for row in rows]

        # Score by embedding similarity if possible
        try:
            from utils.model_loader import embed_single
            from utils.model_loader import cosine_similarity
            context_emb = embed_single(context)

            scored = []
            for entry in entries:
                if entry.embedding is not None:
                    sim = cosine_similarity(context_emb, entry.embedding)
                    score = entry.salience * sim
                else:
                    score = entry.salience * 0.5  # default if no embedding
                scored.append((score, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:top_k]]

        except Exception as e:
            log.warning(f"Embedding-based failure retrieval failed: {e}. Using recency order.")
            return entries[:top_k]

    def decay_salience(self) -> None:
        """
        Apply multiplicative salience decay after a session ends.
        Entries below FAILURE_SALIENCE_MIN are removed.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE failures SET salience = salience * ?",
                (config.FAILURE_SALIENCE_DECAY,),
            )
            conn.execute(
                "DELETE FROM failures WHERE salience < ?",
                (config.FAILURE_SALIENCE_MIN,),
            )
        log.debug("Failure salience decayed.")

    def get_failures_by_obstruction(
        self, obstruction_type: str
    ) -> List[FailureEntry]:
        """Return all failures of a specific obstruction type."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM failures WHERE obstruction_type = ? ORDER BY salience DESC",
                (obstruction_type,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    # -----------------------------------------------------------------------
    # PUBLIC: SESSION CRUD
    # -----------------------------------------------------------------------

    def save_session(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        total_rounds: int,
        termination_reason: str,
        best_conjecture: Optional[str],
        best_lean_coverage: float,
        framing_used: str,
        corpus_subset: List[str],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    start_time,
                    end_time,
                    total_rounds,
                    termination_reason,
                    best_conjecture,
                    best_lean_coverage,
                    framing_used,
                    json.dumps(corpus_subset),
                ),
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row:
            return dict(row)
        return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    # -----------------------------------------------------------------------
    # PUBLIC: FRAMING THOMPSON SAMPLING
    # -----------------------------------------------------------------------

    def sample_framing_thompson(self) -> str:
        """
        Sample a framing variant using Thompson sampling.
        Each framing has a Beta(alpha, beta) distribution representing
        the probability that it leads to Lean proof progress.
        Sample from each distribution and pick the framing with the highest sample.
        """
        import numpy as np

        with self._conn() as conn:
            rows = conn.execute("SELECT framing, alpha, beta FROM framing_stats").fetchall()

        if not rows:
            self._init_framing_stats()
            return config.FRAMING_VARIANTS[0]

        stats = {row["framing"]: (row["alpha"], row["beta"]) for row in rows}
        samples = {}
        for framing in config.FRAMING_VARIANTS:
            alpha, beta = stats.get(framing, (1.0, 1.0))
            samples[framing] = np.random.beta(alpha, beta)

        chosen = max(samples, key=lambda k: samples[k])
        log.info(
            f"Thompson sampling chose framing '{chosen}' "
            f"(samples: {', '.join(f'{k}={v:.3f}' for k,v in samples.items())})"
        )
        return chosen

    def update_framing_stats(self, framing: str, lean_progress_made: bool) -> None:
        """
        Update Thompson sampling parameters for a framing.

        IMPORTANT: Only updated based on Lean proof progress — never based on
        semantic proximity to the target result.

        Args:
            framing:            The framing variant used in the session.
            lean_progress_made: True if any Lean obligations were discharged.
        """
        with self._conn() as conn:
            if lean_progress_made:
                conn.execute(
                    "UPDATE framing_stats SET alpha = alpha + 1.0 WHERE framing = ?",
                    (framing,),
                )
            else:
                conn.execute(
                    "UPDATE framing_stats SET beta = beta + 1.0 WHERE framing = ?",
                    (framing,),
                )
        log.debug(
            f"Framing '{framing}' {'success' if lean_progress_made else 'failure'} recorded."
        )

    def get_framing_stats(self) -> Dict[str, Tuple[float, float]]:
        """Return {framing: (alpha, beta)} for all framings."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT framing, alpha, beta FROM framing_stats"
            ).fetchall()
        return {row["framing"]: (row["alpha"], row["beta"]) for row in rows}

    # -----------------------------------------------------------------------
    # PUBLIC: PAPER QUALITY WEIGHTS (for DPP)
    # -----------------------------------------------------------------------

    def update_paper_quality(
        self,
        source_file: str,
        delta: float = -0.1,
    ) -> None:
        """
        Lower the quality weight of a paper by |delta|.
        Called when a retrieval of this paper correlated with a
        'contradicts_corpus' failure.

        Quality is clipped to [0.1, 1.0].
        NEVER called based on proximity to target result.
        """
        with self._conn() as conn:
            now = datetime.now(timezone.utc).isoformat()
            # Upsert
            conn.execute(
                """
                INSERT INTO paper_quality_weights (source_file, quality, last_updated)
                VALUES (?, 1.0, ?)
                ON CONFLICT(source_file) DO UPDATE SET
                    quality = MAX(0.1, MIN(1.0, quality + ?)),
                    last_updated = ?
                """,
                (source_file, now, delta, now),
            )

    def get_paper_quality_weights(self) -> Dict[str, float]:
        """Return {source_file: quality} for all papers with non-default weights."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_file, quality FROM paper_quality_weights"
            ).fetchall()
        return {row["source_file"]: row["quality"] for row in rows}

    # -----------------------------------------------------------------------
    # PRIVATE: DB INITIALIZATION AND HELPERS
    # -----------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

        # Initialize framing stats if empty
        self._init_framing_stats()

    def _init_framing_stats(self) -> None:
        """Insert all framing variants with uniform Beta prior if not present."""
        with self._conn() as conn:
            existing = {
                row["framing"]
                for row in conn.execute("SELECT framing FROM framing_stats").fetchall()
            }
            for framing in config.FRAMING_VARIANTS:
                if framing not in existing:
                    conn.execute(
                        "INSERT INTO framing_stats (framing, alpha, beta) VALUES (?, ?, ?)",
                        (framing, config.FRAMING_ALPHA_INIT, config.FRAMING_BETA_INIT),
                    )

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for SQLite connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> FailureEntry:
        """Convert a DB row to a FailureEntry."""
        emb = None
        if row["embedding"]:
            try:
                emb = pickle.loads(row["embedding"])
            except Exception:
                emb = None
        return FailureEntry(
            id=row["id"],
            session_id=row["session_id"],
            round_number=row["round_number"],
            timestamp=row["timestamp"],
            assumption_challenged=row["assumption_challenged"],
            technique_attempted=row["technique_attempted"],
            obstruction_type=row["obstruction_type"],
            obstruction_description=row["obstruction_description"],
            proof_step_failed=row["proof_step_failed"],
            branch_summary=row["branch_summary"],
            salience=row["salience"],
            embedding=emb,
        )

    def get_failure_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]

    def get_session_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
