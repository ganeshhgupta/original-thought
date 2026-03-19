"""
config.py — Central configuration for the NOVA Reasoning System.

All tunable parameters live here. Every constant has a comment explaining
what increasing or decreasing it does to system behavior. Do not scatter
magic numbers through the codebase; always import from here.
"""

import os

# ---------------------------------------------------------------------------
# HARDWARE ALLOCATION
# ---------------------------------------------------------------------------

# GPUs for Qwen3-72B generator (tensor parallel across both)
# At Q4_K_M, the 72B model requires ~42GB — split across two 24GB GPUs.
GPU_GENERATOR = [0, 1]

# GPU for DeepSeek-Prover + all tooling (Lean, Z3, SymPy, embeddings)
# The 7B model at BF16 requires ~14GB, leaving ~10GB for tooling.
GPU_CRITIC = 2

# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------

# Primary: GGUF quantized model for llama.cpp (lower memory, faster startup)
QWEN3_MODEL_PATH = "./models/qwen3-72b-instruct-q4_k_m.gguf"

# Fallback: full HF model for vLLM if llama.cpp fails
QWEN3_HF_PATH = "./models/qwen3-72b"

# DeepSeek-Prover for independent formal verification signal
DEEPSEEK_PROVER_PATH = "./models/deepseek-prover-v1.5-rl"

# Small, fast embedding model — intentionally tiny to leave room for DeepSeek
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# CORPUS AND VECTOR STORE
# ---------------------------------------------------------------------------

CORPUS_DIR = "./corpus"
VECTORSTORE_DIR = "./vectorstore"

# Chunk size in tokens for corpus segmentation.
# Larger chunks: more context per retrieval, fewer chunks total.
# Smaller chunks: more precise retrieval, more chunks, slower embedding.
CHUNK_SIZE = 512

# Overlap between adjacent chunks in tokens.
# Higher overlap: less information loss at chunk boundaries.
# Lower overlap: less redundancy, faster embedding.
CHUNK_OVERLAP = 64

# ---------------------------------------------------------------------------
# SESSION PARAMETERS
# ---------------------------------------------------------------------------

# Maximum number of reasoning rounds per session.
# Higher: more thorough exploration. Lower: faster iteration.
MAX_ROUNDS = 30

# Number of Generator reasoning steps before critics engage.
# INCREASING this: Generator explores longer chains before being challenged.
#   Better for finding paths through temporarily unintuitive territory.
#   Worse for efficiency — generator may wander far before getting feedback.
# DECREASING this: Critics engage earlier. More efficient but may kill valid
#   novel chains prematurely before they have established enough scaffolding.
GENERATOR_FREE_STEPS = 12

# Number of samples for self-consistency entropy estimation.
# INCREASING: Better entropy estimate for confidence calibration.
#   Higher inference cost (K * generation time per conjecture).
# DECREASING: Faster but noisier confidence estimate.
#   May mis-rank conjectures due to high-variance entropy measurement.
SELF_CONSISTENCY_K = 5

# Temperature for self-consistency sampling.
# Higher: more diverse samples, higher entropy estimate ceiling.
# Lower: more focused samples, may underestimate true uncertainty.
SELF_CONSISTENCY_TEMP = 0.7

# Number of rounds with no Lean progress before Analogy Agent activates.
# DECREASING: Analogy Agent activates sooner when stalled.
#   Better for problems with cross-domain solutions.
#   Wastes rounds for problems with purely within-domain solutions.
# INCREASING: Saves rounds but misses cross-domain analogies on stalled branches.
ANALOGY_ACTIVATION_ROUND = 3

# Rounds between forced constraint relaxation passes.
# DECREASING: More frequent assumption challenges.
#   Better for finding the right relaxation early.
#   Can fragment the proof attempt if too frequent.
# INCREASING: More focused proof attempts between relaxations.
#   May miss the critical relaxation for many rounds.
CONSTRAINT_RELAXATION_INTERVAL = 3

# Cosine similarity above which consecutive Generator outputs are flagged as degenerate.
# DECREASING: Kills repetitive branches earlier.
#   May terminate valid chains that are converging (high similarity = convergence).
# INCREASING: Allows more repetition before killing.
#   May waste many rounds on truly stuck branches.
DEGENERATE_SIMILARITY_THRESHOLD = 0.92

# Minimum fraction of novel Adversarial Critic objections (below = critics exhausted).
# A session that produces mostly DEGENERATE-tagged objections is stuck.
# DECREASING: Tolerates more critic exhaustion before signaling stall.
# INCREASING: More sensitive to critic depletion — triggers redirect sooner.
CRITIC_NOVELTY_MIN_RATE = 0.10

# Consecutive rounds with zero new Lean obligations discharged = stall signal.
# DECREASING: Triggers redirect sooner after stalling.
#   More responsive but may abandon branches that needed more time.
# INCREASING: More patient with Lean stalls.
#   May waste many rounds on truly stuck branches.
LEAN_FLAT_ROUNDS = 5

# Cosine similarity to a known dead-end branch above which early redirect fires.
# DECREASING: More aggressive early redirect away from dead ends.
#   May redirect prematurely from novel approaches that superficially resemble dead ends.
# INCREASING: Less aggressive. May revisit dead ends.
DEAD_END_SIMILARITY = 0.85

# Number of past objections to track for critic novelty window.
CRITIC_NOVELTY_WINDOW = 5

# ---------------------------------------------------------------------------
# COMPOSITE SCORING WEIGHTS
# These must sum to 1.0.
# ---------------------------------------------------------------------------

# Weight for Lean formal coverage (obligations discharged / total obligations).
# This is the most important signal for scientific validity.
# INCREASING: Prioritizes formally verified content more strongly.
#   This is generally the right direction for mathematical claims.
# DECREASING: Allows high-surprise or high-agreement conjectures to rank
#   above low-coverage but formally verified ones.
SCORE_LEAN_COVERAGE = 0.40

# Weight for conformal uncertainty (self-consistency entropy, inverted).
# Higher score = lower uncertainty = more consistent outputs across samples.
# INCREASING: Rewards claims the model is consistently confident about.
# DECREASING: Less sensitivity to model confidence consistency.
SCORE_CONFORMAL_UNCERTAINTY = 0.30

# Weight for structural surprise (distance from corpus centroid × Lean coverage).
# Measures how far the conjecture is from the "average" of existing literature.
# Higher surprise with high Lean coverage = potentially genuinely novel result.
# INCREASING: Rewards novelty more. Risk: rewards hallucinated novel-sounding claims.
# DECREASING: More conservative. Rewards well-grounded claims.
SCORE_STRUCTURAL_SURPRISE = 0.20

# Weight for cross-model agreement (DeepSeek-Prover endorsement).
# Structural independence: DeepSeek-Prover has different training than Qwen3.
# INCREASING: Rewards claims both models agree on independently.
# DECREASING: Less weight on the second model's opinion.
SCORE_CROSS_MODEL_AGREEMENT = 0.10

# Sanity check — must sum to 1.0
assert abs(SCORE_LEAN_COVERAGE + SCORE_CONFORMAL_UNCERTAINTY +
           SCORE_STRUCTURAL_SURPRISE + SCORE_CROSS_MODEL_AGREEMENT - 1.0) < 1e-9, \
    "Scoring weights must sum to 1.0"

# Upper bound on cluster entropy for uncertainty normalization.
# Derived from K=5 samples — log2(5) ≈ 2.32 bits.
MAX_ENTROPY = 2.322

# ---------------------------------------------------------------------------
# FAILURE MEMORY
# ---------------------------------------------------------------------------

FAILURE_DB_PATH = "./memory/failures.db"

# Multiplicative salience decay applied after each session.
# INCREASING (closer to 1.0): Old failures stay relevant longer.
#   Risk: distant past failures crowd out recent context.
# DECREASING: Old failures fade quickly.
#   Risk: system forgets important obstructions from early sessions.
FAILURE_SALIENCE_DECAY = 0.8

# Minimum salience before a failure entry is pruned from retrieval.
FAILURE_SALIENCE_MIN = 0.1

# Number of most relevant past failures to inject at session start.
# INCREASING: More historical context for the Generator.
#   Risk: context window fills with old failures, less room for current reasoning.
# DECREASING: Lighter context, faster startup, may repeat past mistakes.
FAILURE_RETRIEVAL_TOP_K = 5

# ---------------------------------------------------------------------------
# LEAN 4
# ---------------------------------------------------------------------------

LEAN_WORKSPACE = "./lean_workspace"
LEAN_TIMEOUT_SECONDS = 60
LEAN_MAX_RETRIES = 5
LEAN_CACHE_PATH = "./lean_workspace/cache.json"

# ---------------------------------------------------------------------------
# FRAMING VARIANTS
# Five distinct framings with uniform initial Thompson sampling prior.
# Alpha and beta are updated in the SQLite framing_stats table per session.
# ---------------------------------------------------------------------------

FRAMING_VARIANTS = [
    "limitation",   # Emphasize what is known to fail and why
    "technique",    # Emphasize the proof technique family that might apply
    "analogy",      # Emphasize structural parallels to other domains
    "assumption",   # Emphasize which assumption might be unnecessary
    "compression",  # Emphasize finding a unified explanation for two known results
]

# Initial Thompson sampling parameters (Beta distribution, uniform prior).
FRAMING_ALPHA_INIT = 1.0
FRAMING_BETA_INIT = 1.0

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------

SESSIONS_DIR = "./sessions"
OUTPUTS_DIR = "./outputs"
MEMORY_DIR = "./memory"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ---------------------------------------------------------------------------
# INFERENCE PARAMETERS
# ---------------------------------------------------------------------------

# Generation parameters for the main reasoning pass.
GENERATOR_MAX_TOKENS = 4096
GENERATOR_TEMPERATURE = 0.6

# Generation parameters for critic passes (slightly more focused).
CRITIC_MAX_TOKENS = 2048
CRITIC_TEMPERATURE = 0.3

# Generation parameters for Lean translation (deterministic).
LEAN_TRANSLATION_TEMPERATURE = 0.1
LEAN_TRANSLATION_MAX_TOKENS = 1024

# DeepSeek-Prover evaluation parameters.
DEEPSEEK_MAX_TOKENS = 2048
DEEPSEEK_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# RESOLVE PATHS RELATIVE TO THIS FILE'S LOCATION
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve(path: str) -> str:
    """Resolve a config path relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_BASE_DIR, path)
