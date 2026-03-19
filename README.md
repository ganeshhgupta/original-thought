# original-thought

*can a language model have an original thought?*

not retrieve one. not recombine one. actually have one.

this is an attempt to find out.

---

## the problem

when a mathematician arrives at a novel result, they don't retrieve it. they identify the exact assumption in an existing proof that is doing the most work. they ask what happens if that assumption is relaxed. they notice structural parallels to problems in completely different domains. they follow chains of reasoning through temporarily unintuitive territory before judging whether the chain leads somewhere valid. the discovery emerges from the search process, not from memory.

current LLMs don't do this. they're trained to predict the next token, which means their implicit evaluation signal is distributional similarity. novel ideas score low on this measure. the model's prior is the training distribution. its critic is the training distribution. its generator is the training distribution. the whole system optimizes for familiarity, not truth.

`original-thought` builds an external scaffold that forces a language model to approximate the human reasoning process rather than the retrieval process. then we run it on a real open mathematical problem and see what happens.

---

## the experiment

take a mathematical question at the frontier of transformer expressivity theory. curate a corpus of ~50 papers that are necessary precursors to the answer but don't contain the answer. run the system. compare the output to the actual result (evaluated externally, by hand, after the session). measure not just whether it got there but whether the search process resembled how a human would get there.

the system doesn't know a correct answer exists. it thinks it's working on an open problem. because as far as it knows, it is.

---

## architecture

```
┌────────────────────────────────────────────────────────────┐
│                     session loop                            │
│                                                              │
│  DPP CORPUS SAMPLER                                         │
│  diverse paper subset via determinantal point process        │
│                        │                                     │
│  GENERATOR (Qwen3-72B, GPU 0+1)                             │
│  12 free reasoning steps with tool access:                  │
│  scratchpad · lean_verify · z3_check · sympy · corpus       │
│                        │                                     │
│         ┌──────────────┴──────────────┐                     │
│  ADVERSARIAL CRITIC            DEVIL'S ADVOCATE             │
│  finds weakest point           argues it's already known    │
│         └──────────────┬──────────────┘                     │
│                        │                                     │
│  DEEPSEEK-PROVER-V1.5-RL (GPU 2)                           │
│  independent formal proof evaluation                        │
│                        │                                     │
│  SYNTHESIZER                                                │
│  tags objections · updates scratchpad                       │
│  computes termination signals · decides redirects           │
│                        │                                     │
│  TERMINATION SIGNALS (checked after round 10)               │
│  1. cosine similarity of generator outputs > 0.92           │
│  2. critic novelty rate < 10%                               │
│  3. zero new lean obligations discharged × 5 rounds         │
└────────────────────────────────────────────────────────────┘
```

when 2 of 3 signals fire, the redirect protocol runs:
- **soft:** force constraint relaxation on the most load-bearing assumption
- **medium:** inject a cross-domain analogy from the analogy agent
- **hard:** archive the branch, start fresh with a new thompson-sampled framing

---

## design decisions

**12 free steps before any critic.** a chain that looks wrong at step 3 may be correct at step 12. don't kill ideas too early.

**four structurally independent critics.** Qwen3 (adversarial), Qwen3 (devil's advocate), Lean 4 (formal), DeepSeek-Prover (neural proof). when all four agree, strong signal. when they disagree, the disagreement is informative.

**explicit constraint relaxation.** every assumption the generator treats as load-bearing gets three variants: weakened, negated, replaced. this is the mechanism that produced non-euclidean geometry, special relativity, and most other major discoveries. it should be operationalized, not left implicit.

**analogy agent queries by abstract structure, not topic names.** it finds structurally isomorphic problems in different domains without encoding any knowledge of the target result.

**persistent failure memory.** every failed proof attempt goes into sqlite with a structured taxonomy. at session start, the most relevant past failures get retrieved by embedding similarity and injected into context. salience decays at 0.8x per session so recent failures matter more.

**thompson sampling over framing variants.** five framings: `limitation`, `technique`, `analogy`, `assumption`, `compression`. priors update based on lean obligations discharged. never based on semantic proximity to the target.

**DPP corpus ordering.** maximizes `det(K_S)` over the paper subset at session start. papers associated with `contradicts_corpus` failures get their quality weight lowered.

---

## hardware

```
GPU 0 + 1:  Qwen3-72B-Instruct Q4_K_M (~42GB, tensor parallel)
GPU 2:      DeepSeek-Prover-V1.5-RL BF16 (~14GB)
            + Lean 4 + Mathlib (cpu-bound)
            + Z3 (cpu-bound)
            + SymPy (cpu-bound)
            + all-MiniLM-L6-v2 embeddings (~90MB)
            + ChromaDB
```

---

## install

```bash
git clone https://github.com/ganeshhgupta/original-thought
cd original-thought
python setup.py
```

first run downloads ~56GB of model weights and builds Mathlib (30-90 min, one time). subsequent runs skip everything already present.

```bash
python setup.py --skip-models      # if weights already downloaded
python setup.py --skip-lean        # if lean/mathlib already built
python setup.py --force-rebuild-corpus
```

---

## usage

```bash
# drop papers into corpus/
# write a problem prompt
vim problem_prompt.txt

# run
python main.py --problem ./problem_prompt.txt

# options
python main.py --problem ./problem_prompt.txt --rounds 20
python main.py --problem ./problem_prompt.txt --session-id exp_001
python main.py --problem ./problem_prompt.txt --resume 20241201_143022_a3f8b2c1
```

outputs land in `outputs/{session_id}/results.json` and `outputs/{session_id}/report.md`.

---

## tunable parameters

all in `config.py`:

| parameter | default | what it does |
|---|---|---|
| `GENERATOR_FREE_STEPS` | 12 | steps before critics engage |
| `SELF_CONSISTENCY_K` | 5 | samples for uncertainty estimation |
| `ANALOGY_ACTIVATION_ROUND` | 3 | stall rounds before analogy agent fires |
| `CONSTRAINT_RELAXATION_INTERVAL` | 3 | rounds between assumption relaxation passes |
| `DEGENERATE_SIMILARITY_THRESHOLD` | 0.92 | generator entropy cutoff |
| `LEAN_FLAT_ROUNDS` | 5 | zero-progress rounds before stall signal |
| `SCORE_LEAN_COVERAGE` | 0.40 | weight on formal verification in composite score |
| `FAILURE_SALIENCE_DECAY` | 0.8 | multiplicative salience decay per session |

---

## validity constraints

these are absolute. they're documented in the code. breaking any of them invalidates the experiment.

1. no system prompt, tool description, corpus embedding, failure memory entry, or config value references the specific result the experiment is designed to rediscover.
2. DPP quality weights update only from failure memory signals, never from proximity to the target.
3. thompson sampling parameters update only from lean obligations discharged, never from semantic similarity to the target.
4. the analogy agent queries by abstract structure only, never by technique or paper name.
5. the generator believes it is working on an open problem.
6. session outputs contain no ground-truth field. evaluation happens externally, by hand, after the session ends.

---

## structure

```
original-thought/
├── setup.py
├── main.py
├── config.py
├── problem_prompt.txt
├── corpus/
├── agents/
│   ├── generator.py
│   ├── adversarial_critic.py
│   ├── devils_advocate.py
│   ├── synthesizer.py
│   ├── constraint_relaxer.py
│   └── analogy_agent.py
├── tools/
│   ├── lean_tool.py
│   ├── z3_tool.py
│   ├── sympy_tool.py
│   ├── corpus_retriever.py
│   └── dpp_sampler.py
├── memory/
│   ├── failure_store.py
│   └── scratchpad.py
├── scoring/
│   └── scorer.py
└── utils/
    ├── logger.py
    └── model_loader.py
```

---

## honest limitations

this experiment cannot cleanly distinguish genuine derivation from sophisticated retrieval of adjacent training material. what it can measure is whether the search process resembles human inventive search, whether the system follows the right kind of path even if it arrives at the right answer for the wrong reason, and where exactly the reasoning ceiling is when it fails.

both outcomes are informative. success is interesting. failure is interesting. the process is the point.

---

*models: Qwen3-72B (Alibaba), DeepSeek-Prover-V1.5-RL (DeepSeek AI), Lean 4 / Mathlib (Lean FRO), all-MiniLM-L6-v2 (sentence-transformers)*
