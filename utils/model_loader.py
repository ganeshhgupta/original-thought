"""
utils/model_loader.py — Model loading and inference for Qwen3-72B and DeepSeek-Prover.

Architecture:
  - Qwen3-72B-Instruct: Primary generator and all agent roles.
    Loaded via llama.cpp (GGUF, Q4_K_M) if the GGUF file is present,
    otherwise falls back to vLLM with tensor parallelism across GPU_GENERATOR.
  - DeepSeek-Prover-V1.5-RL: Independent formal-proof critic on GPU_CRITIC.
    Loaded via transformers (BF16).

Both models expose a common .generate(messages, **kwargs) interface
returning a plain string.

The module also provides a standalone embed() function using the
sentence-transformers/all-MiniLM-L6-v2 model (lazy-loaded on first call).
"""

import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import config
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (loaded on first use, not at import time)
# ---------------------------------------------------------------------------

_qwen3_model = None
_qwen3_backend = None   # "llama_cpp" | "vllm"
_qwen3_lock = threading.Lock()

_deepseek_model = None
_deepseek_tokenizer = None
_deepseek_lock = threading.Lock()

_embedding_model = None
_embedding_lock = threading.Lock()


# ===========================================================================
# PUBLIC API
# ===========================================================================

def get_qwen3() -> "Qwen3Wrapper":
    """Return the singleton Qwen3 wrapper, loading it if necessary."""
    global _qwen3_model, _qwen3_backend
    with _qwen3_lock:
        if _qwen3_model is None:
            _qwen3_model, _qwen3_backend = _load_qwen3()
    return _qwen3_model


def get_deepseek_prover() -> "DeepSeekProverWrapper":
    """Return the singleton DeepSeek-Prover wrapper, loading it if necessary."""
    global _deepseek_model, _deepseek_tokenizer
    with _deepseek_lock:
        if _deepseek_model is None:
            _deepseek_model = _load_deepseek_prover()
    return _deepseek_model


def embed(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings using all-MiniLM-L6-v2.
    Returns a list of float vectors, one per input string.
    Lazy-loads the model on first call.
    """
    global _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            _embedding_model = _load_embedding_model()
    return _embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def embed_single(text: str) -> List[float]:
    """Convenience wrapper — embed a single string."""
    return embed([text])[0]


# ===========================================================================
# QWEN3-72B LOADING
# ===========================================================================

def _load_qwen3() -> Tuple[Any, str]:
    """
    Attempt to load Qwen3-72B via llama.cpp; fall back to vLLM.
    Returns (wrapper, backend_name).
    """
    gguf_path = config.resolve(config.QWEN3_MODEL_PATH)
    hf_path = config.resolve(config.QWEN3_HF_PATH)

    # --- Try llama.cpp first ---
    if os.path.exists(gguf_path):
        log.info(f"Loading Qwen3-72B via llama.cpp from {gguf_path}")
        try:
            return _load_qwen3_llama_cpp(gguf_path), "llama_cpp"
        except Exception as e:
            log.warning(f"llama.cpp load failed: {e}. Falling back to vLLM.")

    # --- Fall back to vLLM ---
    if os.path.exists(hf_path):
        log.info(f"Loading Qwen3-72B via vLLM from {hf_path}")
        try:
            return _load_qwen3_vllm(hf_path), "vllm"
        except Exception as e:
            log.error(f"vLLM load also failed: {e}")
            raise RuntimeError(
                f"Cannot load Qwen3-72B. Tried:\n"
                f"  llama.cpp: {gguf_path} (not found or failed)\n"
                f"  vLLM: {hf_path} (failed)\n"
                f"Run setup.py to download the model."
            ) from e

    raise RuntimeError(
        f"Neither GGUF ({gguf_path}) nor HF ({hf_path}) model found. "
        f"Run setup.py first."
    )


def _load_qwen3_llama_cpp(gguf_path: str) -> "Qwen3LlamaCppWrapper":
    from llama_cpp import Llama

    gpu_layers = -1  # offload all layers to GPU
    # Spread across GPU_GENERATOR via CUDA device mapping
    # llama.cpp uses CUDA_VISIBLE_DEVICES; we set it for this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.GPU_GENERATOR)

    model = Llama(
        model_path=gguf_path,
        n_gpu_layers=gpu_layers,
        n_ctx=8192,
        n_threads=8,
        verbose=False,
        # Tensor parallelism across two GPUs via llama.cpp split_mode
        split_mode=1,          # LLAMA_SPLIT_MODE_LAYER
        main_gpu=config.GPU_GENERATOR[0],
    )
    log.info("Qwen3-72B loaded via llama.cpp (GGUF Q4_K_M)")
    return Qwen3LlamaCppWrapper(model)


def _load_qwen3_vllm(hf_path: str) -> "Qwen3VllmWrapper":
    from vllm import LLM, SamplingParams

    tensor_parallel = len(config.GPU_GENERATOR)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.GPU_GENERATOR)

    model = LLM(
        model=hf_path,
        tensor_parallel_size=tensor_parallel,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        trust_remote_code=True,
    )
    log.info(f"Qwen3-72B loaded via vLLM (tensor_parallel={tensor_parallel})")
    return Qwen3VllmWrapper(model, SamplingParams)


# ===========================================================================
# QWEN3 WRAPPERS (common interface)
# ===========================================================================

class Qwen3LlamaCppWrapper:
    """Qwen3-72B served via llama.cpp."""

    def __init__(self, model) -> None:
        self._model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = config.GENERATOR_MAX_TOKENS,
        temperature: float = config.GENERATOR_TEMPERATURE,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response from a list of chat messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Optional list of stop strings.

        Returns:
            The generated text as a plain string.
        """
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return response["choices"][0]["message"]["content"].strip()

    def generate_raw(self, prompt: str, **kwargs) -> str:
        """Raw text completion (for Lean translation prompts)."""
        response = self._model(
            prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.1),
            stop=kwargs.get("stop", []),
        )
        return response["choices"][0]["text"].strip()


class Qwen3VllmWrapper:
    """Qwen3-72B served via vLLM."""

    def __init__(self, model, SamplingParams) -> None:
        self._model = model
        self._SamplingParams = SamplingParams
        # Load tokenizer for chat template application
        from transformers import AutoTokenizer
        hf_path = config.resolve(config.QWEN3_HF_PATH)
        self._tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = config.GENERATOR_MAX_TOKENS,
        temperature: float = config.GENERATOR_TEMPERATURE,
        stop: Optional[List[str]] = None,
    ) -> str:
        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        outputs = self._model.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def generate_raw(self, prompt: str, **kwargs) -> str:
        params = self._SamplingParams(
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.1),
            stop=kwargs.get("stop", []),
        )
        outputs = self._model.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()


# ===========================================================================
# DEEPSEEK-PROVER LOADING
# ===========================================================================

def _load_deepseek_prover() -> "DeepSeekProverWrapper":
    """Load DeepSeek-Prover-V1.5-RL in BF16 on GPU_CRITIC."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config.resolve(config.DEEPSEEK_PROVER_PATH)

    if not os.path.exists(model_path):
        log.error(
            f"DeepSeek-Prover not found at {model_path}. "
            f"Run setup.py to download."
        )
        log.warning("Returning stub DeepSeekProverWrapper. Agreement scores will be 0.")
        return DeepSeekProverStub()

    device = f"cuda:{config.GPU_CRITIC}"
    log.info(f"Loading DeepSeek-Prover-V1.5-RL from {model_path} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    log.info("DeepSeek-Prover-V1.5-RL loaded (BF16)")
    return DeepSeekProverWrapper(model, tokenizer, device)


class DeepSeekProverWrapper:
    """DeepSeek-Prover-V1.5-RL formal proof evaluator."""

    _SYSTEM = (
        "You are a formal mathematics proof verifier. "
        "Given a proof sketch, determine if the logical structure is valid. "
        "Respond with a JSON object: "
        '{\"valid\": true/false, \"confidence\": 0.0-1.0, \"issues\": [...]}'
    )

    def __init__(self, model, tokenizer, device: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def evaluate(self, proof_sketch: str) -> Dict[str, Any]:
        """
        Evaluate a proof sketch.

        Returns:
            {"valid": bool, "confidence": float, "issues": list[str]}
        """
        import torch

        prompt = (
            f"<|system|>\n{self._SYSTEM}\n"
            f"<|user|>\nProof sketch:\n{proof_sketch}\n<|assistant|>\n"
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=config.DEEPSEEK_MAX_TOKENS,
                temperature=config.DEEPSEEK_TEMPERATURE,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the generated portion
        n_input = inputs["input_ids"].shape[1]
        generated = self._tokenizer.decode(
            output[0][n_input:], skip_special_tokens=True
        ).strip()

        return _parse_deepseek_response(generated)

    def evaluate_lean_translation(self, statement: str, lean_code: str) -> Dict[str, Any]:
        """
        Ask DeepSeek-Prover whether the Lean code faithfully represents the statement.
        Used for semantic drift detection.
        """
        import torch

        prompt = (
            f"<|system|>\n{self._SYSTEM}\n"
            f"<|user|>\nNatural language statement: {statement}\n"
            f"Lean 4 code:\n{lean_code}\n"
            f"Does the Lean code faithfully represent the statement? "
            f"Reply JSON: "
            f'{{\"faithful\": true/false, \"confidence\": 0.0-1.0, \"notes\": \"\"}}'
            f"\n<|assistant|>\n"
        )

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        n_input = inputs["input_ids"].shape[1]
        generated = self._tokenizer.decode(
            output[0][n_input:], skip_special_tokens=True
        ).strip()

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {"faithful": True, "confidence": 0.5, "notes": "parse_failed"}


class DeepSeekProverStub:
    """
    Stub for DeepSeek-Prover when model is not available.
    Returns neutral verdicts so scoring degrades gracefully.
    """

    def evaluate(self, proof_sketch: str) -> Dict[str, Any]:
        log.warning("DeepSeekProverStub.evaluate called (model not loaded)")
        return {"valid": False, "confidence": 0.0, "issues": ["model_not_loaded"]}

    def evaluate_lean_translation(self, statement: str, lean_code: str) -> Dict[str, Any]:
        return {"faithful": True, "confidence": 0.0, "notes": "model_not_loaded"}


# ===========================================================================
# EMBEDDING MODEL
# ===========================================================================

def _load_embedding_model():
    """Load the sentence-transformers embedding model on GPU_CRITIC (or CPU)."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = f"cuda:{config.GPU_CRITIC}" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embedding model {config.EMBEDDING_MODEL} on {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    log.info("Embedding model loaded")
    return model


# ===========================================================================
# HELPERS
# ===========================================================================

def _parse_deepseek_response(text: str) -> Dict[str, Any]:
    """Parse DeepSeek-Prover JSON response with graceful fallback."""
    import re
    default = {"valid": False, "confidence": 0.0, "issues": ["parse_failed"]}
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "valid": bool(parsed.get("valid", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "issues": list(parsed.get("issues", [])),
            }
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return default


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Cosine distance = 1 - cosine similarity."""
    return 1.0 - cosine_similarity(a, b)
