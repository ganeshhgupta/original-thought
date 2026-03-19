"""
tools/dpp_sampler.py — Determinantal Point Process corpus ordering.

At the start of each session, the DPP sampler selects an ordered, diverse
subset of corpus paper chunks to present to the Generator.

Why DPP?
  Standard top-k retrieval by query similarity selects a cluster of similar
  papers. DPP sampling selects a DIVERSE subset: papers that cover different
  corners of the embedding space. This prevents the Generator from being
  locked into a single literature cluster at session start.

Algorithm:
  1. Retrieve all paper embeddings from ChromaDB.
  2. Build kernel matrix K where K[i,j] = quality[i] * similarity(i,j) * quality[j].
  3. quality[i] starts at 1.0 for all papers (uniform).
     ONLY update: lower by 0.1 if this paper was retrieved during a round that
     logged a "contradicts_corpus" failure. Never update based on target proximity.
  4. Greedy DPP approximation: maximize det(K_S) via greedy column pivoting.
  5. Order selected subset by citation distance (most-cited papers first).
  6. Return ordered list of chunk IDs.

Quality weights are stored in failure memory DB to persist across sessions.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)


class DPPSampler:
    """
    Greedy Determinantal Point Process sampler for corpus diversity.

    The DPP kernel is: K[i,j] = quality[i] * cos_sim(e_i, e_j) * quality[j]
    where quality weights are derived ONLY from failure memory signals
    (never from proximity to the target result).
    """

    def __init__(self, failure_store=None) -> None:
        """
        Args:
            failure_store: Optional FailureStore instance to read quality weights from.
        """
        self._failure_store = failure_store

    def sample(
        self,
        retriever,
        target_size: int = 20,
        quality_weights: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Sample a diverse subset of corpus chunk IDs via greedy DPP.

        Args:
            retriever:       A CorpusRetriever instance.
            target_size:     How many chunks to select.
            quality_weights: Optional {source_file: quality} dict.
                             If None, loads from failure_store or uses uniform 1.0.

        Returns:
            Ordered list of chunk IDs (by citation distance, most-cited first).
        """
        ids, embeddings, metadatas = retriever.get_all_embeddings()

        if not ids:
            log.warning("DPP sampler: empty corpus. Returning empty list.")
            return []

        n = len(ids)
        target_size = min(target_size, n)

        log.info(f"DPP sampling {target_size} chunks from {n} total corpus chunks.")

        # Build quality vector
        qualities = self._build_quality_vector(metadatas, quality_weights)

        # Build kernel matrix
        K = self._build_kernel(embeddings, qualities)

        # Greedy DPP approximation
        selected_indices = self._greedy_dpp(K, target_size)

        selected_ids = [ids[i] for i in selected_indices]
        selected_metas = [metadatas[i] for i in selected_indices]

        # Order by citation distance
        ordered = self._order_by_citation(
            selected_ids, selected_metas, retriever
        )

        log.info(
            f"DPP selected {len(ordered)} chunks from "
            f"{len(set(m.get('source_file','') for m in selected_metas))} papers."
        )
        return ordered

    def sample_paper_subset(
        self,
        retriever,
        target_papers: int = 10,
        quality_weights: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Sample a diverse subset of PAPERS (not chunks) and return their names,
        citation-ordered.

        Returns:
            List of paper filenames, most-cited first.
        """
        ids, embeddings, metadatas = retriever.get_all_embeddings()
        if not ids:
            return []

        # Aggregate embeddings by paper: use mean embedding per paper
        paper_embeddings: Dict[str, List[List[float]]] = {}
        for emb, meta in zip(embeddings, metadatas):
            src = meta.get("source_file", "unknown")
            paper_embeddings.setdefault(src, []).append(emb)

        papers = list(paper_embeddings.keys())
        n = len(papers)
        target_papers = min(target_papers, n)

        # Mean embedding per paper
        mean_embs = []
        for paper in papers:
            emb_arr = np.array(paper_embeddings[paper])
            mean_emb = emb_arr.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            mean_embs.append(mean_emb.tolist())

        # Quality weights per paper
        qw = quality_weights or self._load_quality_weights()
        qualities = np.array([qw.get(p, 1.0) for p in papers])

        K = self._build_kernel(mean_embs, qualities)
        selected_indices = self._greedy_dpp(K, target_papers)
        selected_papers = [papers[i] for i in selected_indices]

        # Citation order
        return retriever.citation_order(selected_papers)

    # -----------------------------------------------------------------------
    # PRIVATE: KERNEL AND DPP
    # -----------------------------------------------------------------------

    def _build_quality_vector(
        self,
        metadatas: List[Dict],
        quality_weights: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Build quality vector q[i] for each chunk."""
        qw = quality_weights or self._load_quality_weights()
        qualities = np.array([
            qw.get(m.get("source_file", ""), 1.0) for m in metadatas
        ])
        # Clip to [0.1, 1.0]
        qualities = np.clip(qualities, 0.1, 1.0)
        return qualities

    @staticmethod
    def _build_kernel(
        embeddings: List[List[float]],
        qualities: np.ndarray,
    ) -> np.ndarray:
        """
        Build DPP kernel matrix K where K[i,j] = q[i] * cos_sim(e_i, e_j) * q[j].

        Embeddings are assumed to be L2-normalized (from all-MiniLM-L6-v2 output).
        If not normalized, we normalize them here.
        """
        E = np.array(embeddings, dtype=np.float32)

        # Normalize rows
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        E = E / norms

        # Cosine similarity matrix = E @ E^T (since rows are unit vectors)
        S = E @ E.T  # shape: (n, n), values in [-1, 1]

        # Shift to [0, 1] for kernel positivity
        S = (S + 1.0) / 2.0

        # Apply quality weights: K[i,j] = q[i] * S[i,j] * q[j]
        Q = qualities[:, np.newaxis] * qualities[np.newaxis, :]
        K = S * Q

        # Ensure symmetry (numerical errors)
        K = (K + K.T) / 2.0

        return K

    @staticmethod
    def _greedy_dpp(K: np.ndarray, target_size: int) -> List[int]:
        """
        Greedy DPP approximation via column pivoting / greedy maximum determinant.

        At each step, select the item i that maximally increases det(K_S):
          argmax_i log det(K_{S ∪ {i}}) / det(K_S)
          = argmax_i K[i,i] - k_Si^T (K_S)^{-1} k_Si
        where k_Si is the column of K between S and {i}.

        This is the "greedy MAP" DPP approximation (Chen et al. 2018).
        """
        n = K.shape[0]
        selected: List[int] = []
        remaining = list(range(n))

        # Precompute diagonal
        diag_K = np.diag(K).copy()

        # L-ensemble: maintain Cholesky-like update
        # We use the kernel diagonal and conditional variances
        # cond_var[i] = K[i,i] - k_Si^T (K_S)^{-1} k_Si
        # On first step, cond_var = diag(K)

        cond_var = diag_K.copy()
        selected_rows: List[np.ndarray] = []  # k_i vectors projected

        for step in range(target_size):
            if not remaining:
                break

            # Pick the item with maximum conditional variance
            best_idx = remaining[0]
            best_val = -1.0
            for i in remaining:
                if cond_var[i] > best_val:
                    best_val = cond_var[i]
                    best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

            if not remaining:
                break

            # Update conditional variances via rank-1 Cholesky update
            # cond_var[j] -= (K[best_idx, j])^2 / cond_var[best_idx]
            if cond_var[best_idx] < 1e-10:
                break

            e_i = K[best_idx, :]  # full row
            # Project e_i through previous selected to get orthogonal component
            for prev_row in selected_rows:
                e_i = e_i - (np.dot(e_i, prev_row) / np.dot(prev_row, prev_row)) * prev_row

            scale = np.sqrt(max(cond_var[best_idx], 1e-10))
            e_i_normalized = e_i / scale
            selected_rows.append(e_i_normalized)

            for j in remaining:
                dot = np.dot(e_i_normalized, K[best_idx, :])
                update = (K[best_idx, j] ** 2) / max(cond_var[best_idx], 1e-10)
                cond_var[j] = max(0.0, cond_var[j] - update)

        return selected

    # -----------------------------------------------------------------------
    # PRIVATE: CITATION ORDERING
    # -----------------------------------------------------------------------

    @staticmethod
    def _order_by_citation(
        chunk_ids: List[str],
        metadatas: List[Dict],
        retriever,
    ) -> List[str]:
        """
        Reorder selected chunks so chunks from more-cited papers come first.
        Within a paper, preserve original chunk order.
        """
        # Get paper citation order
        paper_names = list(set(m.get("source_file", "") for m in metadatas))
        ordered_papers = retriever.citation_order(paper_names)
        paper_rank = {p: i for i, p in enumerate(ordered_papers)}

        # Sort chunks: primary key = paper citation rank, secondary = chunk index
        chunk_order = []
        for cid, meta in zip(chunk_ids, metadatas):
            src = meta.get("source_file", "")
            rank = paper_rank.get(src, len(ordered_papers))
            cidx = int(meta.get("chunk_index", 0))
            chunk_order.append((rank, cidx, cid))

        chunk_order.sort()
        return [cid for _, _, cid in chunk_order]

    # -----------------------------------------------------------------------
    # PRIVATE: QUALITY WEIGHTS
    # -----------------------------------------------------------------------

    def _load_quality_weights(self) -> Dict[str, float]:
        """
        Load per-paper quality weights from failure memory.
        Papers associated with 'contradicts_corpus' failures have reduced quality.
        Quality weights are NEVER updated based on target proximity.
        """
        if self._failure_store is None:
            return {}
        try:
            return self._failure_store.get_paper_quality_weights()
        except Exception as e:
            log.warning(f"Could not load quality weights from failure store: {e}")
            return {}
