"""
tools/corpus_retriever.py — ChromaDB vector store query interface.

Manages the corpus embedding pipeline and provides semantic retrieval
for the Generator and Analogy Agent.

Pipeline:
  1. On first run (or when new files appear), scan corpus/ directory.
  2. Extract text from PDFs (pdfplumber first, PyMuPDF fallback).
  3. Chunk each document into 512-token overlapping segments (64-token overlap).
  4. Embed all chunks using sentence-transformers/all-MiniLM-L6-v2.
  5. Store in ChromaDB at vectorstore/.
  6. On subsequent runs, check modification timestamps and only re-embed changed files.

Query interface:
  retrieve(query, top_k=5)  ->  list of CorpusChunk

IMPORTANT: Query strings must be algebraic/structural descriptions, NOT topic names.
The Analogy Agent specifically must query by structure to prevent it from directly
encoding knowledge of the target result.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

import config
from utils.logger import get_logger, progress_bar

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Token approximation (without loading a real tokenizer for chunking)
# ---------------------------------------------------------------------------

def _approx_token_count(text: str) -> int:
    """Approximate token count: ~4 characters per token (rough but fast)."""
    return max(1, len(text) // 4)


def _chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Chunk text into overlapping segments of approximately chunk_size tokens.
    Uses word boundaries to avoid splitting mid-word.
    """
    words = text.split()
    if not words:
        return []

    # Convert chunk_size/overlap from tokens to words (4 chars/token ≈ 1.3 words/token)
    words_per_chunk = max(10, int(chunk_size * 1.3))
    words_per_overlap = max(5, int(overlap * 1.3))
    step = words_per_chunk - words_per_overlap

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(path: str) -> str:
    """
    Extract text from a PDF file.
    Tries pdfplumber first; falls back to PyMuPDF (fitz).
    Raises RuntimeError if both fail.
    """
    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if pages:
                return "\n\n".join(pages)
            # Empty extraction — fall through to PyMuPDF
    except ImportError:
        log.warning("pdfplumber not installed, trying PyMuPDF")
    except Exception as e:
        log.warning(f"pdfplumber failed on {path}: {e}. Trying PyMuPDF.")

    # Try PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = [doc[i].get_text("text") for i in range(len(doc))]
        doc.close()
        text = "\n\n".join(p for p in pages if p.strip())
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"PyMuPDF failed on {path}: {e}.")

    # Try PyPDF2 as last resort
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            text = "\n\n".join(pages)
            if text.strip():
                return text
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"PyPDF2 failed on {path}: {e}.")

    raise RuntimeError(
        f"All PDF extraction methods failed for: {path}. "
        f"Install pdfplumber and/or pymupdf: pip install pdfplumber pymupdf"
    )


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

@dataclass
class CorpusChunk:
    """A single retrieved corpus chunk with metadata."""
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    distance: float      # lower = more similar
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "distance": self.distance,
        }


class CorpusRetriever:
    """
    ChromaDB-backed corpus retriever.

    Handles embedding, indexing, and semantic search over the paper corpus.
    """

    TIMESTAMP_FILE = ".last_embed_timestamp.json"
    COLLECTION_NAME = "nova_corpus"

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        vectorstore_dir: Optional[str] = None,
    ) -> None:
        self.corpus_dir = Path(config.resolve(corpus_dir or config.CORPUS_DIR))
        self.vectorstore_dir = Path(config.resolve(vectorstore_dir or config.VECTORSTORE_DIR))
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.vectorstore_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._timestamp_file = self.vectorstore_dir / self.TIMESTAMP_FILE
        self._timestamps: Dict[str, float] = self._load_timestamps()

    # -----------------------------------------------------------------------
    # PUBLIC: BUILD / UPDATE INDEX
    # -----------------------------------------------------------------------

    def build_or_update_index(self, force_rebuild: bool = False) -> int:
        """
        Scan corpus dir, embed new/changed files, update ChromaDB.

        Returns:
            Number of new chunks added.
        """
        corpus_files = self._scan_corpus()
        if not corpus_files:
            log.warning(f"No documents found in {self.corpus_dir}. Add PDFs or TXTs.")
            return 0

        new_files = self._find_new_files(corpus_files, force_rebuild)

        if not new_files:
            log.info("Corpus index is up to date. No re-embedding needed.")
            return 0

        log.info(f"Embedding {len(new_files)} new/changed files...")

        total_chunks_added = 0
        for file_path in progress_bar(new_files, desc="Embedding corpus", total=len(new_files)):
            try:
                chunks_added = self._embed_file(file_path)
                total_chunks_added += chunks_added
                self._timestamps[str(file_path)] = file_path.stat().st_mtime
            except Exception as e:
                log.error(f"Failed to embed {file_path}: {e}")
                # Continue with other files — do not abort entire indexing run

        self._save_timestamps()
        log.info(
            f"Index build complete. {total_chunks_added} chunks added. "
            f"Total collection size: {self._collection.count()}"
        )
        return total_chunks_added

    # -----------------------------------------------------------------------
    # PUBLIC: RETRIEVAL
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_source: Optional[str] = None,
    ) -> List[CorpusChunk]:
        """
        Retrieve top_k most semantically similar chunks for the query.

        Args:
            query:         The search query (should be an algebraic/structural
                           description, not a topic name).
            top_k:         Number of chunks to return.
            filter_source: If provided, only return chunks from this source file.

        Returns:
            List of CorpusChunk objects, ordered by similarity (closest first).
        """
        from utils.model_loader import embed

        if self._collection.count() == 0:
            log.warning("Corpus index is empty. Build index first.")
            return []

        query_embedding = embed([query])[0]

        where = None
        if filter_source:
            where = {"source_file": filter_source}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as e:
            log.error(f"ChromaDB query failed: {e}")
            return []

        chunks = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            embs = results.get("embeddings", [[]])[0] if results.get("embeddings") else [None] * len(docs)

            for doc, meta, dist, emb in zip(docs, metas, dists, embs):
                chunks.append(CorpusChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    source_file=meta.get("source_file", ""),
                    page_number=meta.get("page_number"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                    distance=float(dist),
                    embedding=list(emb) if emb is not None else None,
                ))

        return chunks

    def get_all_embeddings(self) -> Tuple[List[str], List[List[float]], List[Dict]]:
        """
        Retrieve all embeddings from the collection.
        Returns (ids, embeddings, metadatas).
        Used by the DPP sampler.
        """
        count = self._collection.count()
        if count == 0:
            return [], [], []

        results = self._collection.get(
            include=["embeddings", "metadatas"],
            limit=count,
        )
        return (
            results["ids"],
            [list(e) for e in results["embeddings"]],
            results["metadatas"],
        )

    def get_corpus_centroid(self) -> Optional[List[float]]:
        """
        Compute the mean embedding vector across the entire corpus.
        Used by the scorer for structural surprise computation.
        """
        ids, embeddings, _ = self.get_all_embeddings()
        if not embeddings:
            return None

        import numpy as np
        emb_array = np.array(embeddings)
        centroid = emb_array.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.tolist()

    def get_paper_names(self) -> List[str]:
        """Return list of unique source file names in the index."""
        _, _, metas = self.get_all_embeddings()
        return list(set(m.get("source_file", "") for m in metas))

    def citation_order(self, paper_names: List[str]) -> List[str]:
        """
        Order papers by citation frequency (papers cited by more other papers go first).
        Simple heuristic: count how many other papers' text contains the filename stem.

        Returns the reordered list.
        """
        # Load all chunk texts for citation counting
        count = self._collection.count()
        if count == 0:
            return paper_names

        results = self._collection.get(include=["documents", "metadatas"], limit=count)
        all_texts = results["documents"]

        citation_counts: Dict[str, int] = {}
        for name in paper_names:
            stem = Path(name).stem.lower()
            # Count how many chunks from OTHER papers mention this paper's stem
            cnt = sum(
                1 for doc, meta in zip(all_texts, results["metadatas"])
                if stem in doc.lower() and meta.get("source_file", "") != name
            )
            citation_counts[name] = cnt

        return sorted(paper_names, key=lambda n: citation_counts.get(n, 0), reverse=True)

    # -----------------------------------------------------------------------
    # PRIVATE: FILE SCANNING AND EMBEDDING
    # -----------------------------------------------------------------------

    def _scan_corpus(self) -> List[Path]:
        """Find all .pdf and .txt files in corpus_dir."""
        files = []
        if not self.corpus_dir.exists():
            log.warning(f"Corpus directory does not exist: {self.corpus_dir}")
            return files
        for suffix in (".pdf", ".txt"):
            files.extend(self.corpus_dir.glob(f"**/*{suffix}"))
        return sorted(files)

    def _find_new_files(
        self, all_files: List[Path], force_rebuild: bool
    ) -> List[Path]:
        """Return files that are new or have been modified since last embedding."""
        if force_rebuild:
            return all_files
        new = []
        for f in all_files:
            last_mtime = self._timestamps.get(str(f), 0.0)
            current_mtime = f.stat().st_mtime
            if current_mtime > last_mtime:
                new.append(f)
        return new

    def _embed_file(self, file_path: Path) -> int:
        """
        Extract text, chunk, embed, and store one file.
        Returns number of chunks added.
        """
        from utils.model_loader import embed as embed_fn

        # Extract text
        if file_path.suffix.lower() == ".pdf":
            text = _extract_pdf_text(str(file_path))
        else:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                text = f.read()

        if not text.strip():
            log.warning(f"Empty content extracted from {file_path}")
            return 0

        chunks = _chunk_text(text)
        if not chunks:
            return 0

        log.debug(f"  {file_path.name}: {len(chunks)} chunks")

        # Embed all chunks at once
        embeddings = embed_fn(chunks)

        # Prepare metadata
        ids = []
        documents = []
        metadatas = []
        embs_to_add = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = self._make_chunk_id(str(file_path), i)

            # Remove existing chunk if present (handles re-indexing)
            try:
                self._collection.delete(ids=[chunk_id])
            except Exception:
                pass

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source_file": file_path.name,
                "source_path": str(file_path),
                "chunk_index": i,
                "chunk_id": chunk_id,
                "page_number": None,
            })
            embs_to_add.append(emb)

        # Add in batches of 100 to avoid memory issues
        batch_size = 100
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            self._collection.add(
                ids=ids[batch_start:batch_end],
                documents=documents[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
                embeddings=embs_to_add[batch_start:batch_end],
            )

        return len(chunks)

    @staticmethod
    def _make_chunk_id(file_path: str, chunk_index: int) -> str:
        h = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"{h}_{chunk_index:05d}"

    def _load_timestamps(self) -> Dict[str, float]:
        if self._timestamp_file.exists():
            try:
                with open(self._timestamp_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_timestamps(self) -> None:
        with open(self._timestamp_file, "w") as f:
            json.dump(self._timestamps, f, indent=2)
