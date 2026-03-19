NOVA REASONING SYSTEM — CORPUS DIRECTORY
==========================================

Place your scientific papers and documents here BEFORE running the system.

SUPPORTED FORMATS:
  - PDF files (.pdf)   — parsed with pdfplumber, fallback to PyMuPDF
  - Plain text (.txt)  — read directly

HOW THE CORPUS IS USED:
  1. On first run (or when new files appear), all documents are chunked into
     512-token overlapping segments (64-token overlap) and embedded using
     sentence-transformers/all-MiniLM-L6-v2.
  2. Embeddings are stored persistently in ../vectorstore/ (ChromaDB).
  3. During each session, the DPP sampler selects a diverse subset of papers
     to present to the Generator in a citation-ordered sequence.
  4. The Analogy Agent and Generator make live queries into the vectorstore
     using structural algebraic descriptions (not topic names).

WHAT TO PUT HERE (for the Strassen-attention experiment):
  - Impossibility results for one-layer transformers
    (Peng et al. 2024, Sanford et al. 2023, etc.)
  - Attention mechanism foundational papers (Vaswani et al. 2017)
  - Bilinear complexity / tensor rank literature
    (Strassen 1969, Blaser surveys, Pan 1980, Coppersmith-Winograd, etc.)
  - Circuit complexity and communication complexity foundations
  - Papers on fast matrix multiplication algorithms
  - Papers on expressivity of transformers
  - Any relevant survey papers on the mathematics of neural sequence models

WHAT NOT TO PUT HERE:
  - Any paper containing the specific mathematical result the system
    is attempting to rediscover. The experiment requires that the target
    result is NOT in the corpus. The corpus should contain all logical
    PRECURSORS to the target result, but not the result itself.
  - Experimental results that directly answer the open question.

CORPUS INTEGRITY:
  Adding the wrong papers can invalidate the experiment. The system has
  no knowledge of what the "correct answer" is, so it cannot filter
  inappropriate papers automatically. This is the experimenter's responsibility.

FILES ADDED INCREMENTALLY:
  The system checks file modification timestamps on every run.
  Only new or modified files are re-embedded. You can safely add papers
  at any time between sessions without full re-indexing.
