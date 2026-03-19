"""
scoring/scorer.py — Multi-signal conjecture scorer.

At session termination, all surviving conjectures are scored and ranked
using four structurally independent signals:

  1. Lean Coverage (weight 0.40):
     Fraction of proof obligations discharged by the Lean 4 compiler.
     This is the most important signal for scientific validity.
     Higher Lean coverage = more formally verified content.

  2. Self-Consistency Uncertainty (weight 0.30):
     Generate the conjecture K=5 times at temperature 0.7.
     Embed all K outputs. Compute cluster entropy.
     Lower entropy = more consistent = higher score.
     Rationale: claims the model consistently generates have lower uncertainty.

  3. Structural Surprise (weight 0.20):
     Cosine distance from the corpus centroid × Lean coverage weight.
     Farther from corpus average = more potentially novel.
     Multiplied by Lean coverage: high surprise with low coverage = probably wrong.
     Rationale: a novel claim far from existing literature that is also formally
     verified is more interesting than a novel claim with no verification.

  4. Cross-Model Agreement (weight 0.10):
     Does DeepSeek-Prover independently agree the proof sketch is valid?
     DeepSeek has different training data and architecture from Qwen3.
     When both agree, the signal is stronger than either alone.

IMPORTANT INVARIANT: The scorer does not compare to any target result.
Scoring is purely based on internal quality signals. Comparison to the
target happens externally, after the session, by the human experimenter.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ConjectureScore:
    """Full scoring breakdown for one conjecture."""
    name: str
    statement: str
    proof_sketch: str

    # Raw component values
    lean_coverage: float = 0.0
    lean_obligations_discharged: int = 0
    lean_obligations_total: int = 0

    cluster_entropy: float = 0.0
    uncertainty_score: float = 0.0  # 1.0 - normalized entropy

    structural_surprise: float = 0.0
    weighted_surprise: float = 0.0   # surprise * lean_coverage

    deepseek_valid: bool = False
    deepseek_confidence: float = 0.0
    cross_model_agreement: float = 0.0

    # Composite
    composite_score: float = 0.0

    # Additional metadata
    contributing_failures: List[str] = field(default_factory=list)
    contributing_analogies: List[str] = field(default_factory=list)
    lean_proof_fragments: List[str] = field(default_factory=list)
    z3_constraints_satisfied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "statement": self.statement,
            "proof_sketch": self.proof_sketch,
            "lean_coverage": self.lean_coverage,
            "lean_obligations_discharged": self.lean_obligations_discharged,
            "lean_obligations_total": self.lean_obligations_total,
            "cluster_entropy": self.cluster_entropy,
            "uncertainty_score": self.uncertainty_score,
            "structural_surprise": self.structural_surprise,
            "weighted_surprise": self.weighted_surprise,
            "deepseek_valid": self.deepseek_valid,
            "deepseek_confidence": self.deepseek_confidence,
            "cross_model_agreement": self.cross_model_agreement,
            "composite_score": self.composite_score,
            "contributing_failures": self.contributing_failures,
            "contributing_analogies": self.contributing_analogies,
            "lean_proof_fragments": self.lean_proof_fragments,
            "z3_constraints_satisfied": self.z3_constraints_satisfied,
        }


class ConjectureScorer:
    """
    Multi-signal conjecture scorer.

    Each method computes one scoring component. The final score is the
    weighted sum of all components.
    """

    def score_all(
        self,
        conjectures: Dict[str, Dict[str, Any]],
        deepseek_prover,
        corpus_retriever=None,
    ) -> List[ConjectureScore]:
        """
        Score all conjectures and return them sorted by composite score (highest first).

        Args:
            conjectures:     Dict from scratchpad.conjectures.
            deepseek_prover: DeepSeekProverWrapper for cross-model evaluation.
            corpus_retriever: CorpusRetriever for centroid computation (optional).

        Returns:
            List of ConjectureScore objects, sorted highest to lowest.
        """
        if not conjectures:
            log.warning("No conjectures to score.")
            return []

        log.info(f"Scoring {len(conjectures)} conjectures...")

        # Compute corpus centroid once (reused for all conjectures)
        corpus_centroid = None
        if corpus_retriever:
            try:
                corpus_centroid = corpus_retriever.get_corpus_centroid()
            except Exception as e:
                log.warning(f"Could not compute corpus centroid: {e}")

        scores = []
        for name, entry in conjectures.items():
            try:
                score = self._score_one(
                    name=name,
                    entry=entry,
                    deepseek_prover=deepseek_prover,
                    corpus_centroid=corpus_centroid,
                )
                scores.append(score)
            except Exception as e:
                log.error(f"Failed to score conjecture '{name}': {e}")
                # Include with zero score rather than dropping
                scores.append(ConjectureScore(
                    name=name,
                    statement=entry.get("statement", ""),
                    proof_sketch=entry.get("proof_sketch", ""),
                    composite_score=0.0,
                ))

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        log.info(
            f"Scoring complete. Top score: {scores[0].composite_score:.3f} "
            f"({scores[0].name})"
        )
        return scores

    def _score_one(
        self,
        name: str,
        entry: Dict[str, Any],
        deepseek_prover,
        corpus_centroid: Optional[List[float]],
    ) -> ConjectureScore:
        """Score a single conjecture."""
        statement = entry.get("statement", "")
        proof_sketch = entry.get("proof_sketch", "")

        score = ConjectureScore(name=name, statement=statement, proof_sketch=proof_sketch)

        # ---------------------------------------------------------------
        # Component 1: Lean Coverage
        # ---------------------------------------------------------------
        lean_result = entry.get("lean_result", {})
        if lean_result:
            discharged = lean_result.get("obligations_discharged", 0)
            remaining = lean_result.get("obligations_remaining", 0)
            total = discharged + remaining
        else:
            # Fall back to stored lean_coverage fraction
            discharged = 0
            remaining = 0
            total = 0

        lean_coverage_stored = entry.get("lean_coverage", 0.0)
        if total > 0:
            score.lean_coverage = discharged / total
        else:
            score.lean_coverage = lean_coverage_stored

        score.lean_obligations_discharged = discharged
        score.lean_obligations_total = max(total, 1)

        # Store Lean proof fragment if available
        if lean_result and lean_result.get("lean_code"):
            score.lean_proof_fragments = [lean_result["lean_code"]]

        log.debug(f"  {name}: lean_coverage={score.lean_coverage:.3f}")

        # ---------------------------------------------------------------
        # Component 2: Self-Consistency Uncertainty
        # ---------------------------------------------------------------
        score.cluster_entropy, score.uncertainty_score = \
            self._compute_self_consistency(statement, proof_sketch)
        log.debug(f"  {name}: entropy={score.cluster_entropy:.3f}, uncertainty_score={score.uncertainty_score:.3f}")

        # ---------------------------------------------------------------
        # Component 3: Structural Surprise
        # ---------------------------------------------------------------
        if corpus_centroid:
            score.structural_surprise = self._compute_structural_surprise(
                statement, corpus_centroid
            )
        else:
            score.structural_surprise = 0.5  # neutral when centroid unavailable
        score.weighted_surprise = score.structural_surprise * score.lean_coverage
        log.debug(f"  {name}: surprise={score.structural_surprise:.3f}, weighted={score.weighted_surprise:.3f}")

        # ---------------------------------------------------------------
        # Component 4: Cross-Model Agreement
        # ---------------------------------------------------------------
        try:
            deepseek_result = deepseek_prover.evaluate(proof_sketch)
            score.deepseek_valid = deepseek_result.get("valid", False)
            score.deepseek_confidence = deepseek_result.get("confidence", 0.0)
            score.cross_model_agreement = 1.0 if score.deepseek_valid else 0.0
        except Exception as e:
            log.warning(f"DeepSeek evaluation failed for {name}: {e}")
            score.cross_model_agreement = 0.0
        log.debug(f"  {name}: deepseek_valid={score.deepseek_valid}")

        # ---------------------------------------------------------------
        # Composite Score
        # ---------------------------------------------------------------
        score.composite_score = (
            config.SCORE_LEAN_COVERAGE       * score.lean_coverage +
            config.SCORE_CONFORMAL_UNCERTAINTY * score.uncertainty_score +
            config.SCORE_STRUCTURAL_SURPRISE  * score.weighted_surprise +
            config.SCORE_CROSS_MODEL_AGREEMENT * score.cross_model_agreement
        )

        return score

    # -----------------------------------------------------------------------
    # PRIVATE: INDIVIDUAL COMPONENT COMPUTATIONS
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_self_consistency(
        statement: str,
        proof_sketch: str,
    ) -> Tuple[float, float]:
        """
        Generate the conjecture K times at temperature SELF_CONSISTENCY_TEMP.
        Embed all outputs. Compute cluster entropy.
        Returns (cluster_entropy, uncertainty_score).

        Higher entropy = more uncertain = lower score.
        """
        from utils.model_loader import get_qwen3, embed

        prompt = (
            f"State the following mathematical conjecture in one precise sentence, "
            f"then give a one-sentence proof sketch:\n\n{statement}\n\nProof sketch: {proof_sketch}"
        )
        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = []
        try:
            qwen3 = get_qwen3()
            for _ in range(config.SELF_CONSISTENCY_K):
                out = qwen3.generate(
                    messages,
                    max_tokens=256,
                    temperature=config.SELF_CONSISTENCY_TEMP,
                )
                outputs.append(out)
        except Exception as e:
            log.warning(f"Self-consistency generation failed: {e}")
            return 1.0, 0.5  # maximum uncertainty if generation fails

        if len(outputs) < 2:
            return 1.0, 0.5

        try:
            embeddings = embed(outputs)
            entropy = _compute_embedding_cluster_entropy(embeddings)
            uncertainty_score = 1.0 - min(entropy / config.MAX_ENTROPY, 1.0)
            return entropy, uncertainty_score
        except Exception as e:
            log.warning(f"Entropy computation failed: {e}")
            return 1.0, 0.5

    @staticmethod
    def _compute_structural_surprise(
        statement: str,
        corpus_centroid: List[float],
    ) -> float:
        """
        Compute cosine distance from the corpus centroid.
        Returns value in [0, 1] where 1 = maximally different from corpus average.
        """
        from utils.model_loader import embed_single, cosine_distance
        try:
            stmt_embedding = embed_single(statement)
            surprise = cosine_distance(stmt_embedding, corpus_centroid)
            return float(max(0.0, min(1.0, surprise)))
        except Exception as e:
            log.warning(f"Structural surprise computation failed: {e}")
            return 0.5  # neutral


# ---------------------------------------------------------------------------
# Entropy computation helpers
# ---------------------------------------------------------------------------

def _compute_embedding_cluster_entropy(embeddings: List[List[float]]) -> float:
    """
    Compute entropy of a set of embeddings using k-means clustering.

    Algorithm:
      1. Cluster K embeddings into min(K, 3) clusters via cosine k-means.
      2. Compute cluster assignment probabilities.
      3. Compute Shannon entropy of the distribution.

    Lower entropy = all embeddings cluster tightly = model is consistent.
    Higher entropy = embeddings spread across clusters = model is uncertain.
    """
    import numpy as np

    n = len(embeddings)
    if n < 2:
        return 0.0

    # Normalize embeddings
    emb = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb = emb / norms

    # k-means with k = min(n, 3) clusters
    k = min(n, 3)

    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(emb)
    except ImportError:
        # Manual k-means approximation using pairwise cosine similarity
        labels = _simple_cluster(emb, k)

    # Compute cluster size distribution
    counts = np.bincount(labels, minlength=k)
    probs = counts / counts.sum()

    # Shannon entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return float(entropy)


def _simple_cluster(emb: Any, k: int) -> Any:
    """
    Simple greedy clustering when sklearn is unavailable.
    Assigns each embedding to the nearest seed (seeds = first k embeddings).
    """
    import numpy as np
    seeds = emb[:k]
    labels = []
    for e in emb:
        sims = seeds @ e  # cosine similarity (normalized embeddings)
        labels.append(int(np.argmax(sims)))
    return np.array(labels)
