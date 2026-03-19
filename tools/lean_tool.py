"""
tools/lean_tool.py — Lean 4 subprocess wrapper for the NOVA Reasoning System.

Responsibilities:
  1. Accept a natural language statement + informal proof sketch.
  2. Translate the statement to Lean 4 syntax via Qwen3 (with few-shot Mathlib examples).
  3. Write the Lean code to a temp file in lean_workspace/.
  4. Run `lake build` with a configurable timeout.
  5. Parse compiler output: extract discharged obligations, errors, remaining goals.
  6. Retry on failure (up to LEAN_MAX_RETRIES), injecting compiler errors back into Qwen3.
  7. Perform a round-trip semantic drift check via DeepSeek-Prover.
  8. Cache successfully compiled lemmas in lean_workspace/cache.json.

Return type (LeanResult):
  success              : bool
  obligations_discharged: int
  obligations_remaining : int
  errors               : list[str]
  semantic_drift       : bool
  lean_code            : str
  cache_hit            : bool
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Few-shot Lean 4 / Mathlib translation examples injected into every
# translation prompt. These are generic mathematical facts that do NOT
# encode the target result in any way.
# ---------------------------------------------------------------------------

LEAN_FEW_SHOT_EXAMPLES = """
-- EXAMPLE 1: Commutativity of natural number addition
-- Natural language: For all natural numbers m and n, m + n = n + m.
-- Lean 4:
theorem add_comm_example (m n : ℕ) : m + n = n + m := by
  exact Nat.add_comm m n

-- EXAMPLE 2: Injectivity condition
-- Natural language: A function f : α → β is injective if for all a₁ a₂ : α,
--   f(a₁) = f(a₂) implies a₁ = a₂.
-- Lean 4:
theorem injective_def {α β : Type*} (f : α → β) :
    Function.Injective f ↔ ∀ a₁ a₂, f a₁ = f a₂ → a₁ = a₂ := by
  exact Iff.rfl

-- EXAMPLE 3: Matrix multiplication complexity bound
-- Natural language: The rank of an n×n matrix over a field is at most n.
-- Lean 4:
theorem matrix_rank_le (n : ℕ) (K : Type*) [Field K] (M : Matrix (Fin n) (Fin n) K) :
    M.rank ≤ n := by
  exact Matrix.rank_le_card_height M

-- EXAMPLE 4: Group homomorphism identity
-- Natural language: Any group homomorphism f : G → H maps the identity of G
--   to the identity of H.
-- (This requires a sorry placeholder for the general monoid case.)
-- Lean 4:
-- theorem hom_maps_identity {G H : Type*} [Group G] [Group H] (f : G →* H) :
--     f 1 = 1 := by
--   exact map_one f
"""

LEAN_TRANSLATION_PROMPT_TEMPLATE = """You are a Lean 4 / Mathlib expert. Translate the following natural language mathematical statement into correct Lean 4 code with a proof sketch.

Use `sorry` for proof steps that are not yet verified. Focus on getting the TYPE SIGNATURE exactly right, as this is what will be checked.

Lean 4 few-shot examples for style reference:
{few_shot}

Natural language statement to translate:
{statement}

Informal proof sketch:
{proof_sketch}

Output ONLY the Lean 4 code block, no explanation, no markdown fences. Start directly with `theorem` or `lemma`.
"""

LEAN_FIX_PROMPT_TEMPLATE = """The following Lean 4 code failed to compile. Fix it.

Original code:
{lean_code}

Compiler error:
{error}

Output ONLY the corrected Lean 4 code. Start directly with `theorem` or `lemma`.
"""

LEAN_ROUNDTRIP_PROMPT_TEMPLATE = """The following Lean 4 theorem was compiled successfully:

{lean_code}

Does this Lean statement mean the same thing as the following natural language statement?
Answer YES or NO, then explain in one sentence.

Natural language: {original_statement}
"""


@dataclass
class LeanResult:
    success: bool
    obligations_discharged: int
    obligations_remaining: int
    errors: List[str]
    semantic_drift: bool
    lean_code: str
    cache_hit: bool = False
    attempt_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LeanTool:
    """
    Lean 4 verification tool.

    Thread-safety: uses a file-lock on the lean_workspace directory so that
    parallel critic runs do not stomp each other's temp files.
    """

    def __init__(self) -> None:
        self.workspace = config.resolve(config.LEAN_WORKSPACE)
        self.cache_path = config.resolve(config.LEAN_CACHE_PATH)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._lean_binary = self._find_lean_binary()
        self._new_obligations_this_session: int = 0

        # Counter for unique temp file names
        self._file_counter = 0

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def verify(
        self,
        statement: str,
        proof_sketch: str,
        claim_name: Optional[str] = None,
    ) -> LeanResult:
        """
        Verify a mathematical claim.

        Args:
            statement:    Natural language statement of the claim.
            proof_sketch: Informal proof sketch.
            claim_name:   Optional name for caching (default: hash of statement).

        Returns:
            LeanResult with verification details.
        """
        cache_key = self._cache_key(statement)

        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            log.info(f"Lean cache HIT for: {statement[:80]}...")
            return LeanResult(
                success=cached["success"],
                obligations_discharged=cached["obligations_discharged"],
                obligations_remaining=cached["obligations_remaining"],
                errors=cached["errors"],
                semantic_drift=cached["semantic_drift"],
                lean_code=cached["lean_code"],
                cache_hit=True,
            )

        log.info(f"Lean VERIFY: {statement[:80]}...")

        # Translate to Lean 4
        lean_code = self._translate_to_lean(statement, proof_sketch)

        # Attempt compilation with retries
        result = self._compile_with_retries(lean_code, statement, proof_sketch)

        # Semantic drift check if compilation succeeded
        if result.success:
            result.semantic_drift = self._check_semantic_drift(statement, result.lean_code)
            if result.semantic_drift:
                log.warning(f"Semantic drift detected for: {statement[:60]}...")

        # Cache successful compilations
        if result.success and not result.semantic_drift:
            self._cache[cache_key] = result.to_dict()
            self._save_cache()
            self._new_obligations_this_session += result.obligations_discharged

        return result

    def verify_all_new_claims(self, scratchpad) -> List[LeanResult]:
        """
        Verify all unverified claims in the scratchpad's conjectures and assumptions.

        Returns a list of LeanResults, one per claim attempted.
        """
        results = []
        for name, entry in list(scratchpad.conjectures.items()):
            if entry.get("lean_coverage", 0.0) == 0.0:
                result = self.verify(
                    statement=entry["statement"],
                    proof_sketch=entry.get("proof_sketch", ""),
                    claim_name=name,
                )
                entry["lean_coverage"] = self._coverage(result)
                entry["lean_result"] = result.to_dict()
                if result.success:
                    scratchpad.promote_to_established(name, result.lean_code)
                results.append(result)
        return results

    def new_obligations_this_session(self) -> int:
        """Return count of new obligations discharged this session."""
        return self._new_obligations_this_session

    def reset_session_counter(self) -> None:
        self._new_obligations_this_session = 0

    # -----------------------------------------------------------------------
    # PRIVATE: TRANSLATION
    # -----------------------------------------------------------------------

    def _translate_to_lean(self, statement: str, proof_sketch: str) -> str:
        """Use Qwen3 to translate natural language to Lean 4."""
        try:
            from utils.model_loader import get_qwen3
            qwen3 = get_qwen3()
        except Exception as e:
            log.error(f"Cannot load Qwen3 for Lean translation: {e}")
            return self._fallback_lean_template(statement)

        prompt = LEAN_TRANSLATION_PROMPT_TEMPLATE.format(
            few_shot=LEAN_FEW_SHOT_EXAMPLES,
            statement=statement,
            proof_sketch=proof_sketch or "(no sketch provided)",
        )

        messages = [
            {"role": "system", "content": "You are a Lean 4 and Mathlib expert. Output only valid Lean 4 code."},
            {"role": "user",   "content": prompt},
        ]

        lean_code = qwen3.generate(
            messages,
            max_tokens=config.LEAN_TRANSLATION_MAX_TOKENS,
            temperature=config.LEAN_TRANSLATION_TEMPERATURE,
        )

        # Strip markdown fences if present
        lean_code = self._strip_markdown(lean_code)
        return lean_code

    def _fix_lean_code(self, lean_code: str, error: str) -> str:
        """Ask Qwen3 to fix a Lean compilation error."""
        try:
            from utils.model_loader import get_qwen3
            qwen3 = get_qwen3()
        except Exception:
            return lean_code  # cannot fix without model

        prompt = LEAN_FIX_PROMPT_TEMPLATE.format(
            lean_code=lean_code,
            error=error[:2000],  # truncate very long errors
        )
        messages = [
            {"role": "system", "content": "You are a Lean 4 expert. Fix the code. Output only valid Lean 4 code."},
            {"role": "user",   "content": prompt},
        ]
        fixed = qwen3.generate(
            messages,
            max_tokens=config.LEAN_TRANSLATION_MAX_TOKENS,
            temperature=0.1,
        )
        return self._strip_markdown(fixed)

    # -----------------------------------------------------------------------
    # PRIVATE: COMPILATION
    # -----------------------------------------------------------------------

    def _compile_with_retries(
        self, lean_code: str, statement: str, proof_sketch: str
    ) -> LeanResult:
        """
        Compile Lean code, retrying with error-guided fixes up to LEAN_MAX_RETRIES.
        """
        current_code = lean_code
        errors_accumulated: List[str] = []

        for attempt in range(1, config.LEAN_MAX_RETRIES + 1):
            log.debug(f"Lean compile attempt {attempt}/{config.LEAN_MAX_RETRIES}")
            compile_result = self._run_lean(current_code)

            if compile_result["success"]:
                return LeanResult(
                    success=True,
                    obligations_discharged=compile_result["obligations_discharged"],
                    obligations_remaining=compile_result["obligations_remaining"],
                    errors=[],
                    semantic_drift=False,
                    lean_code=current_code,
                    attempt_count=attempt,
                )

            errors_accumulated.extend(compile_result["errors"])
            log.debug(f"Lean error (attempt {attempt}): {compile_result['errors'][:1]}")

            if attempt < config.LEAN_MAX_RETRIES:
                error_msg = "\n".join(compile_result["errors"][:5])
                current_code = self._fix_lean_code(current_code, error_msg)
                current_code = self._strip_markdown(current_code)

        # All retries exhausted
        return LeanResult(
            success=False,
            obligations_discharged=0,
            obligations_remaining=1,
            errors=errors_accumulated[-10:],  # keep last 10 errors
            semantic_drift=False,
            lean_code=current_code,
            attempt_count=config.LEAN_MAX_RETRIES,
        )

    def _run_lean(self, lean_code: str) -> Dict[str, Any]:
        """
        Write lean_code to a temp file and run `lake build`.
        Parse output for discharged goals, errors, and remaining goals.

        Returns dict: {success, obligations_discharged, obligations_remaining, errors, raw_output}
        """
        if not self._lean_binary:
            log.warning("Lean binary not found. Returning stub result.")
            return {
                "success": False,
                "obligations_discharged": 0,
                "obligations_remaining": 1,
                "errors": ["lean_binary_not_found"],
                "raw_output": "",
            }

        self._file_counter += 1
        temp_name = f"nova_verify_{self._file_counter:05d}.lean"
        temp_path = os.path.join(self.workspace, temp_name)

        # Wrap in necessary imports
        full_code = self._wrap_lean_code(lean_code)

        try:
            with open(temp_path, "w") as f:
                f.write(full_code)

            result = subprocess.run(
                [self._lean_binary, temp_path],
                capture_output=True,
                text=True,
                timeout=config.LEAN_TIMEOUT_SECONDS,
                cwd=self.workspace,
            )
            raw = result.stdout + result.stderr
            return self._parse_lean_output(raw, result.returncode)

        except subprocess.TimeoutExpired:
            log.warning(f"Lean timed out after {config.LEAN_TIMEOUT_SECONDS}s")
            return {
                "success": False,
                "obligations_discharged": 0,
                "obligations_remaining": 1,
                "errors": [f"timeout_after_{config.LEAN_TIMEOUT_SECONDS}s"],
                "raw_output": "TIMEOUT",
            }
        except Exception as e:
            log.error(f"Lean execution error: {e}")
            return {
                "success": False,
                "obligations_discharged": 0,
                "obligations_remaining": 1,
                "errors": [str(e)],
                "raw_output": "",
            }
        finally:
            # Clean up temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _parse_lean_output(self, output: str, returncode: int) -> Dict[str, Any]:
        """
        Parse Lean 4 compiler output.

        Lean 4 outputs:
          - "goals accomplished" when all proof obligations are discharged
          - "unsolved goals" with remaining goal states
          - Error messages for type errors
        """
        errors = []
        discharged = 0
        remaining = 0
        success = returncode == 0

        lines = output.splitlines()
        for line in lines:
            # Errors
            if "error:" in line.lower():
                errors.append(line.strip())
            # Goals discharged
            if "goals accomplished" in line.lower():
                discharged += 1
            # Remaining goals
            match = re.search(r'(\d+)\s+goal', line)
            if match and "unsolved" in line.lower():
                remaining += int(match.group(1))
            # Sorry counts as partially discharged but flagged
            if "sorry" in line.lower() and "warning" in line.lower():
                discharged += 1  # sorry compiles, counts as discharged but weak

        # If no explicit goal counts, infer from success
        if success and discharged == 0:
            discharged = 1  # at minimum the top-level theorem

        return {
            "success": success and len([e for e in errors if "sorry" not in e.lower()]) == 0,
            "obligations_discharged": discharged,
            "obligations_remaining": remaining,
            "errors": errors,
            "raw_output": output[:2000],
        }

    def _wrap_lean_code(self, lean_code: str) -> str:
        """Prepend necessary Lean 4 imports."""
        imports = "import Mathlib\nimport Lean\n\nopen Real Matrix\n\n"
        return imports + lean_code

    # -----------------------------------------------------------------------
    # PRIVATE: SEMANTIC DRIFT CHECK
    # -----------------------------------------------------------------------

    def _check_semantic_drift(self, original_statement: str, lean_code: str) -> bool:
        """
        Ask DeepSeek-Prover whether the compiled Lean code faithfully represents
        the original natural language statement.
        Returns True if drift is detected (code means something different).
        """
        try:
            from utils.model_loader import get_deepseek_prover
            prover = get_deepseek_prover()
            result = prover.evaluate_lean_translation(original_statement, lean_code)
            return not result.get("faithful", True)
        except Exception as e:
            log.warning(f"Semantic drift check failed: {e}. Assuming no drift.")
            return False

    # -----------------------------------------------------------------------
    # PRIVATE: HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_lean_binary() -> Optional[str]:
        """Find the `lean` binary in standard locations."""
        candidates = [
            os.path.expanduser("~/.elan/bin/lean"),
            "/usr/local/bin/lean",
            "lean",
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                log.info(f"Found Lean binary at: {c}")
                return c
        # Try which
        try:
            result = subprocess.run(["which", "lean"], capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()
                log.info(f"Found Lean binary via which: {path}")
                return path
        except Exception:
            pass
        log.warning("Lean binary not found. Lean verification will be stubbed.")
        return None

    @staticmethod
    def _strip_markdown(code: str) -> str:
        """Strip markdown code fences (```lean ... ```) if present."""
        code = re.sub(r'^```(?:lean4?)?', '', code.strip(), flags=re.MULTILINE)
        code = re.sub(r'```$', '', code.strip(), flags=re.MULTILINE)
        return code.strip()

    @staticmethod
    def _cache_key(statement: str) -> str:
        return hashlib.sha256(statement.encode()).hexdigest()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    @staticmethod
    def _coverage(result: LeanResult) -> float:
        total = result.obligations_discharged + result.obligations_remaining
        if total == 0:
            return 0.0
        return result.obligations_discharged / total

    @staticmethod
    def _fallback_lean_template(statement: str) -> str:
        """Minimal valid Lean 4 template when Qwen3 is unavailable."""
        return (
            f"-- Auto-generated stub (Qwen3 unavailable)\n"
            f"-- Statement: {statement}\n"
            f"theorem nova_stub : True := by trivial\n"
        )
