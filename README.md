# NOVA Reasoning System

**Neural Observation and Verification Architecture for Scientific Discovery**

A multi-agent mathematical reasoning system designed to answer one of the hardest open questions in AI: *can a language model generate genuinely novel scientific ideas through a reasoning process that resembles how humans actually arrive at discoveries, rather than through pattern retrieval from training data?*

---

## The Core Problem

When a human mathematician arrives at a novel result, they do not retrieve it from memory. They perform a sequence of cognitive operations that have been studied carefully: they identify the exact assumption in existing proofs that is doing the most work; they ask what happens if that assumption is relaxed; they notice structural parallels to problems in different domains; they follow chains of reasoning through temporarily unintuitive territory before judging whether the chain leads somewhere valid; and they are productively disturbed by inconsistencies that others have ignored. The discovery is not in their memory — it emerges from the search process.

Current large language models do not do this. They are trained to predict the next token, which means their implicit evaluation signal is distributional similarity. When a language model encounters a novel idea, it computes something like *"how often does this type of claim appear in contexts where the surrounding argument is valid?"* Novel ideas score low on this measure. The model's prior is the training distribution. Its critic is the training distribution. Its generator is the training distribution. The entire system is optimizing for familiarity, not for truth.

NOVA is an attempt to build an external scaffold that forces a language model to approximate the human reasoning process rather than the retrieval process.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                     NOVA SESSION LOOP                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DPP CORPUS SAMPLER                                  │  │
│  │  Diverse paper subset via Determinantal Point Process│  │
│  └─────────────────────────┬────────────────────────────┘  │
│                             │ citation-ordered paper subset  │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │  GENERATOR (Qwen3-72B, GPU 0+1)                       │  │
│  │  12 free reasoning steps with tool access:            │  │
│  │  • scratchpad_write/read/list  • lean_verify          │  │
│  │  • z3_check                    • sympy_compute        │  │
│  │  • corpus_retrieve                                    │  │
│  └──────────┬──────────────────────────┬─────────────────┘  │
│             │                          │                      │
│  ┌──────────▼─────────┐   ┌───────────▼──────────────────┐  │
│  │  ADVERSARIAL CRITIC│   │  DEVIL'S ADVOCATE            │  │
│  │  Finds weakest     │   │  Argues work is unnecessary   │  │
│  │  point in proof    │   │  or already known             │  │
│  └──────────┬─────────┘   └───────────┬──────────────────┘  │
│             │                          │                      │
│  ┌──────────▼──────────────────────────▼──────────────────┐  │
│  │  DEEPSEEK-PROVER-V1.5-RL (GPU 2)                       │  │
│  │  Independent formal proof evaluation                   │  │
│  └──────────┬─────────────────────────────────────────────┘  │
│             │                                                  │
│  ┌──────────▼─────────────────────────────────────────────┐  │
│  │  SYNTHESIZER                                           │  │
│  │  Tags objections • Updates scratchpad                  │  │
│  │  Computes termination signals • Decides redirects      │  │
│  └──────────┬─────────────────────────────────────────────┘  │
│             │                                                  │
│  ┌──────────▼─────────────────────────────────────────────┐  │
│  │  TERMINATION SIGNALS (checked after round 10)          │  │
│  │  1. Generator entropy (cosine similarity > 0.92)       │  │
│  │  2. Critic novelty rate (< 10% novel objections)       │  │
│  │  3. Lean progress rate (0 new obligations × 5 rounds)  │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

When 2 of 3 signals fire, the **Redirect Protocol** executes:
- **Layer 1 (soft):** Force constraint relaxation on the most load-bearing assumption.
- **Layer 2 (medium):** Inject a cross-domain analogy from the Analogy Agent.
- **Layer 3 (hard):** Archive the branch to failure memory, start fresh with a new Thompson-sampled framing.

---

## Key Design Principles

### 1. Separation of Generation from Evaluation
The Generator runs for **12 steps completely unchallenged** before any critic engages. This replicates the human ability to follow a chain of reasoning through temporarily unintuitive territory before judging it. A chain that looks wrong at step 3 may be correct at step 12.

### 2. Structurally Independent Critics
Four critics with different error distributions:
- **Qwen3-72B (Adversarial Critic):** Finds the weakest technical point in the proof
- **Qwen3-72B (Devil's Advocate):** Argues the work is unnecessary
- **Lean 4 + Mathlib:** Formal proof verification
- **DeepSeek-Prover-V1.5-RL:** Independent neural proof evaluation

When all four agree, the signal is strong. When they disagree, the disagreement is informative.

### 3. Operationalized Constraint Relaxation
The system explicitly identifies every assumption the Generator is treating as load-bearing and proposes variants where each assumption is:
- **Weakened:** what if it only holds approximately or for a subclass?
- **Negated:** what if it is false — does the impossibility proof survive?
- **Replaced:** what other assumption could substitute while preserving the key properties?

This replicates the human mechanism that produced non-Euclidean geometry, special relativity, and numerous other discoveries.

### 4. Operationalized Cross-Domain Analogy
The Analogy Agent queries the corpus using an **abstract algebraic structure description**, not topic names. This prevents it from encoding knowledge of the target result. It finds structurally isomorphic problems in different domains and proposes whether the same resolution technique might apply here.

### 5. Persistent Failure Memory
Every failed proof attempt is stored in SQLite with a structured taxonomy:
- `logical_gap`, `missing_lemma`, `contradicts_corpus`, `precision_issue`, `bilinear_structure`, `dimension_mismatch`

At the start of each session, the most relevant past failures are retrieved by embedding similarity and injected into the Generator's context. Salience decays multiplicatively (`FAILURE_SALIENCE_DECAY = 0.8`) so recent failures matter more than ancient ones.

### 6. Thompson Sampling for Framing
Five framing variants are maintained with Beta-distributed Thompson sampling parameters:
- `limitation` — start from what is known to fail
- `technique` — emphasize which proof technique family might apply
- `analogy` — emphasize structural parallels to other domains
- `assumption` — emphasize which assumption might be unnecessary
- `compression` — find a unified explanation for two known results

Framing priors are updated based on **Lean proof obligations discharged**, never based on semantic proximity to the target result.

### 7. DPP Corpus Diversity
At each session start, the Determinantal Point Process sampler selects a diverse subset of corpus papers (maximizing `det(K_S)` where `K[i,j] = quality[i] × similarity(i,j) × quality[j]`). Papers associated with `contradicts_corpus` failures have their quality weight lowered. This prevents the system from being locked into a single literature cluster at session start.

---

## Hardware Requirements

```
GPU 0 + GPU 1:  Qwen3-72B-Instruct at Q4_K_M quantization (~42GB)
                Tensor parallel via llama.cpp (fallback: vLLM)

GPU 2:          DeepSeek-Prover-V1.5-RL at BF16 (~14GB)
                + Lean 4 server (CPU-bound)
                + Z3 solver (CPU-bound)
                + SymPy (CPU-bound)
                + all-MiniLM-L6-v2 embeddings (~90MB)
                + ChromaDB SQLite
```

Minimum: 3 GPUs with 24GB VRAM each. 4 GPUs available but only 3 active.

---

## Installation

```bash
# Clone or place the nova_reasoning/ directory on your system
cd nova_reasoning/

# Run setup (idempotent — safe to re-run)
python setup.py

# Options:
python setup.py --skip-models      # if models already downloaded
python setup.py --skip-lean        # if Lean/Mathlib already built
python setup.py --force-rebuild-corpus  # force re-embedding corpus
```

**First run will:**
1. Install system packages (cmake, build tools, etc.)
2. Create `./venv/` with all Python dependencies
3. Install Lean 4 via elan + build Mathlib (30-90 minutes, one time)
4. Download Qwen3-72B-Instruct GGUF (~42GB) from HuggingFace
5. Download DeepSeek-Prover-V1.5-RL (~14GB) from HuggingFace
6. Embed corpus documents into ChromaDB

**Subsequent runs skip everything already present.**

---

## Usage

### 1. Prepare the Corpus

Add scientific papers to `./corpus/` before the first run:
```
corpus/
├── README.txt              ← instructions
├── vaswani2017attention.pdf
├── peng2024impossibility.pdf
├── sanford2023one_layer.pdf
├── strassen1969gaussian.pdf
├── blaser2003bilinear.pdf
└── ...
```

See `corpus/README.txt` for guidance on what to include (and critically, what NOT to include to preserve experimental validity).

### 2. Write a Problem Prompt

Create a plain text file describing the mathematical question:
```
vim problem_prompt.txt
```

See `problem_prompt.txt` (auto-generated by setup.py) for an example format.

### 3. Run a Session

```bash
# Basic run
python main.py --problem ./problem_prompt.txt

# Custom number of rounds
python main.py --problem ./problem_prompt.txt --rounds 20

# Named session
python main.py --problem ./problem_prompt.txt --session-id myexp_001

# Resume an interrupted session
python main.py --problem ./problem_prompt.txt --resume 20241201_143022_a3f8b2c1
```

### 4. Read the Output

```
outputs/{session_id}/
├── results.json    ← machine-readable full results
└── report.md       ← human-readable session report
```

---

## Output Format

### `results.json`
```json
{
  "session_id": "...",
  "timestamp": "...",
  "rounds_completed": 18,
  "termination_reason": "all_signals_fired",
  "problem_prompt": "...",
  "framing_used": "assumption",
  "corpus_files_used": ["strassen1969.pdf", ...],
  "conjectures": [
    {
      "rank": 1,
      "statement": "...",
      "proof_sketch": "...",
      "lean_coverage": 0.71,
      "lean_obligations_discharged": 5,
      "lean_obligations_total": 7,
      "lean_proof_fragments": ["theorem nova_001 ..."],
      "deepseek_prover_verdict": "valid",
      "composite_score": 0.684,
      ...
    }
  ],
  "failure_memory_entries_added": 12,
  "redirect_events": [...],
  "session_log": [...]
}
```

### `report.md`
A human-readable Markdown report with:
- Problem statement
- Round-by-round summary paragraphs
- Top-ranked conjecture stated in natural language
- Lean proof fragment (if obligations discharged)
- Most frequently retrieved corpus papers
- Influential past failure entries
- Specific obstructions encountered and how they were addressed
- What remains unproven and why

---

## Tunable Parameters

All parameters are in `config.py` with comments explaining the effect of each change:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GENERATOR_FREE_STEPS` | 12 | Steps before critics engage. Higher = longer free exploration. |
| `SELF_CONSISTENCY_K` | 5 | Samples for uncertainty estimation. Higher = better calibration. |
| `ANALOGY_ACTIVATION_ROUND` | 3 | Stall rounds before cross-domain analogy fires. |
| `CONSTRAINT_RELAXATION_INTERVAL` | 3 | Rounds between assumption relaxation passes. |
| `DEGENERATE_SIMILARITY_THRESHOLD` | 0.92 | Generator entropy cutoff. Higher = more tolerant of repetition. |
| `LEAN_FLAT_ROUNDS` | 5 | Consecutive zero-progress rounds before stall signal. |
| `SCORE_LEAN_COVERAGE` | 0.40 | Weight on formal verification in composite score. |
| `FAILURE_SALIENCE_DECAY` | 0.8 | Multiplicative salience decay per session. |

---

## Experimental Validity Invariants

The following constraints are **absolute** and documented in the codebase:

1. No system prompt, tool description, corpus embedding, failure memory entry, or configuration value contains any reference to the specific result the experiment is designed to rediscover.

2. The DPP quality weights are only updated based on failure memory signals. Never based on proximity to the target.

3. Thompson sampling parameters are only updated based on Lean obligations discharged. Never based on semantic similarity to the target.

4. The Analogy Agent queries the corpus using abstract structural descriptions derived from the current proof obligation. It is never given the name of any technique, paper, or result that directly solves the problem.

5. The Generator is told it is working on an open problem. It does not know that a correct answer exists.

6. Session output files contain no ground-truth field. Evaluation against the target happens externally, after the session, by the human experimenter.

---

## Project Structure

```
nova_reasoning/
├── setup.py                    # Idempotent installation script
├── main.py                     # Entry point, session runner
├── config.py                   # All configuration constants
├── problem_prompt.txt          # Example problem (auto-generated)
├── .gitignore
├── README.md
│
├── corpus/                     # Drop PDFs/TXTs here
│   └── README.txt
├── models/                     # Downloaded weights (gitignored)
├── vectorstore/                # ChromaDB persistent store (gitignored)
├── lean_workspace/             # Lean 4 + Mathlib project
├── sessions/                   # Per-session checkpoint JSON
├── outputs/                    # Final results and reports
│   └── {session_id}/
│       ├── results.json
│       └── report.md
│
├── agents/
│   ├── generator.py            # Primary reasoning agent
│   ├── adversarial_critic.py   # Proof weakness finder
│   ├── devils_advocate.py      # Novelty challenger
│   ├── synthesizer.py          # Round manager + termination logic
│   ├── constraint_relaxer.py   # Assumption variant generator
│   └── analogy_agent.py        # Cross-domain structure matcher
│
├── tools/
│   ├── lean_tool.py            # Lean 4 subprocess + caching
│   ├── z3_tool.py              # Z3 SMT solver API
│   ├── sympy_tool.py           # Symbolic algebra
│   ├── corpus_retriever.py     # ChromaDB query interface
│   └── dpp_sampler.py          # DPP corpus ordering
│
├── memory/
│   ├── failure_store.py        # SQLite failure memory
│   └── scratchpad.py           # In-session working memory
│
├── scoring/
│   └── scorer.py               # Multi-signal conjecture ranking
│
└── utils/
    ├── logger.py               # Structured session logging
    └── model_loader.py         # Qwen3 + DeepSeek-Prover loaders
```

---

## Scientific Background

This system is designed around a specific class of mathematical questions in the theory of neural sequence models — specifically, the question of whether known impossibility results for one-layer transformers survive when their implicit precision assumptions are removed, and whether architectural modifications can provably overcome them.

The seed corpus should contain:
- Impossibility results for one-layer transformers
- The attention mechanism paper (Vaswani et al. 2017)
- Bilinear complexity and tensor rank literature (Strassen 1969, Blaser surveys)
- Fast matrix multiplication algorithms
- Circuit complexity and communication complexity foundations
- Expressivity results for transformers

The seed corpus should NOT contain any paper that directly answers the open question. The experiment requires the system to reason from precursors to the result, not retrieve the result itself.

---

## Citation and Attribution

If you use NOVA in research, please cite the constituent model papers:
- **Qwen3-72B-Instruct:** Qwen Team, Alibaba Group
- **DeepSeek-Prover-V1.5-RL:** DeepSeek AI
- **Lean 4 / Mathlib:** The Lean FRO and Mathlib community
- **all-MiniLM-L6-v2:** sentence-transformers / Microsoft Research

---

## License

Research use. See model licenses for Qwen3, DeepSeek-Prover, Lean 4, and Mathlib.
