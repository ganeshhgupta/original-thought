"""
setup.py — One-shot setup and dependency installer for the NOVA Reasoning System.

Run this ONCE before any session. It is fully idempotent: it checks for
each dependency before installing and skips steps already done.

What this script does:
  1. System packages (apt-get)
  2. Python virtual environment + pip packages
  3. Lean 4 + elan installation
  4. Lean workspace initialization + Mathlib build
  5. Download Qwen3-72B-Instruct (GGUF Q4_K_M or HF fallback)
  6. Download DeepSeek-Prover-V1.5-RL
  7. Build corpus vector store (ChromaDB + all-MiniLM-L6-v2)

Usage:
  python setup.py [--skip-models] [--skip-lean] [--force-rebuild-corpus]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = True, env: dict = None, cwd: str = None) -> int:
    """Run a shell command, printing it first. Returns exit code."""
    print(f"\n$ {cmd}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd, shell=True, env=full_env, cwd=cwd
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result.returncode


def run_capture(cmd: str) -> str:
    """Run a command and return stdout, or '' on failure."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def heading(text: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}\n")


def check_exists(path: str) -> bool:
    return Path(path).exists()


def check_command(cmd: str) -> bool:
    """Return True if a command is available on PATH."""
    return shutil.which(cmd) is not None


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_DIR = PROJECT_ROOT / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_PIP = VENV_DIR / "bin" / "pip"


def pip_install(packages: str, extra_args: str = "") -> None:
    """Install packages into the virtual environment."""
    run(f"{VENV_PIP} install --upgrade {extra_args} {packages}")


# ===========================================================================
# STEP 1: System dependencies
# ===========================================================================

def install_system_deps() -> None:
    heading("STEP 1: System Dependencies")

    if not check_command("cmake"):
        print("Installing system packages...")
        run(
            "sudo apt-get update && sudo apt-get install -y "
            "git curl wget build-essential cmake ninja-build "
            "python3-pip python3-dev libssl-dev zlib1g-dev libbz2-dev "
            "libreadline-dev libsqlite3-dev libffi-dev liblzma-dev "
            "libgmp-dev libmpfr-dev pkg-config"
        )
    else:
        print("System packages already installed (cmake found). Skipping.")


# ===========================================================================
# STEP 2: Python virtual environment
# ===========================================================================

def setup_python_env() -> None:
    heading("STEP 2: Python Virtual Environment")

    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR}...")
        run(f"{sys.executable} -m venv {VENV_DIR}")
    else:
        print(f"Virtual environment already exists at {VENV_DIR}. Checking packages...")

    # Core packages — check each before installing to save time
    packages = [
        # PyTorch with CUDA 12.1
        ("torch", "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"),
        # Transformers stack
        ("transformers", "transformers accelerate bitsandbytes sentencepiece"),
        # vLLM for serving Qwen3-72B with tensor parallelism
        ("vllm", "vllm"),
        # Embeddings
        ("sentence_transformers", "sentence-transformers"),
        # Vector store
        ("chromadb", "chromadb"),
        # Constraint solver
        ("z3", "z3-solver"),
        # Symbolic math
        ("sympy", "sympy"),
        # PDF parsing (all three backends)
        ("pdfplumber", "pdfplumber pymupdf"),
        ("PyPDF2", "PyPDF2"),
        # Numerics
        ("numpy", "numpy scipy scikit-learn"),
        # Progress / display
        ("tqdm", "tqdm rich colorama"),
        # Geometric (for DPP)
        ("sklearn", ""),  # already installed via scikit-learn
        # llama-cpp for GGUF loading
        ("llama_cpp", "llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"),
        # HuggingFace CLI
        ("huggingface_hub", "huggingface-hub"),
    ]

    for import_name, install_spec in packages:
        if not install_spec:
            continue
        # Check if already importable
        check_result = subprocess.run(
            [str(VENV_PYTHON), "-c", f"import {import_name}"],
            capture_output=True
        )
        if check_result.returncode == 0:
            print(f"  {import_name}: already installed")
        else:
            print(f"  {import_name}: installing...")
            run(f"{VENV_PIP} install {install_spec}", check=False)


# ===========================================================================
# STEP 3: Lean 4 + Mathlib
# ===========================================================================

def setup_lean(skip: bool = False) -> None:
    heading("STEP 3: Lean 4 and Mathlib")

    if skip:
        print("Skipping Lean installation (--skip-lean flag).")
        return

    elan_bin = Path.home() / ".elan" / "bin" / "elan"
    lean_bin = Path.home() / ".elan" / "bin" / "lean"

    # Install elan (Lean version manager)
    if not elan_bin.exists():
        print("Installing elan (Lean version manager)...")
        run(
            "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
            "-sSf | sh -s -- -y"
        )
        # Re-source environment
        elan_env_script = Path.home() / ".elan" / "env"
        if elan_env_script.exists():
            os.environ["PATH"] = str(Path.home() / ".elan" / "bin") + ":" + os.environ.get("PATH", "")
    else:
        print("elan already installed.")

    if not lean_bin.exists():
        print("Installing Lean 4 (stable)...")
        elan_path = str(Path.home() / ".elan" / "bin" / "elan")
        run(f"{elan_path} install leanprover/lean4:stable", check=False)
    else:
        print("Lean 4 already installed.")

    # Set up Lean workspace
    workspace = PROJECT_ROOT / "lean_workspace"
    workspace.mkdir(exist_ok=True)
    lakefile = workspace / "lakefile.lean"

    if not (workspace / "lake-manifest.json").exists():
        print("Initializing Lean workspace...")
        lean_elan_path = str(Path.home() / ".elan" / "bin")
        env_with_lean = {
            **os.environ,
            "PATH": lean_elan_path + ":" + os.environ.get("PATH", ""),
        }

        # Initialize lake project
        init_rc = subprocess.run(
            f"{lean_elan_path}/lake init lean_workspace",
            shell=True, cwd=str(workspace), env=env_with_lean
        ).returncode

        # Write lakefile with Mathlib dependency
        lakefile_content = '''import Lake
open Lake DSL

package «lean_workspace» where
  -- add any package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib «LeanWorkspace» where
  -- add any library configuration options here
'''
        with open(lakefile, "w") as f:
            f.write(lakefile_content)

        # Also write the .lean default file
        default_lean = workspace / "LeanWorkspace.lean"
        with open(default_lean, "w") as f:
            f.write("-- NOVA Lean Workspace\nimport Mathlib\n")

    print("Running lake update (downloads Mathlib — this may take 5-20 minutes)...")
    lean_elan_path = str(Path.home() / ".elan" / "bin")
    env_with_lean = {
        **os.environ,
        "PATH": lean_elan_path + ":" + os.environ.get("PATH", ""),
    }

    update_rc = subprocess.run(
        f"{lean_elan_path}/lake update",
        shell=True, cwd=str(workspace), env=env_with_lean
    ).returncode

    if update_rc == 0:
        print("lake update complete.")
        print("Building Mathlib (this will take 30-90 minutes the first time)...")
        print("Progress will be shown by lake build.")
        build_rc = subprocess.run(
            f"{lean_elan_path}/lake build",
            shell=True, cwd=str(workspace), env=env_with_lean
        ).returncode
        if build_rc == 0:
            print("Mathlib build complete.")
        else:
            print(f"Warning: lake build returned {build_rc}. Lean verification may be limited.")
    else:
        print(f"Warning: lake update returned {update_rc}. Lean setup may be incomplete.")

    # Create Lean cache file
    cache_path = workspace / "cache.json"
    if not cache_path.exists():
        with open(cache_path, "w") as f:
            json.dump({}, f)
        print("Lean cache file initialized.")


# ===========================================================================
# STEP 4: Download Qwen3-72B-Instruct
# ===========================================================================

def download_qwen3(skip: bool = False) -> None:
    heading("STEP 4: Qwen3-72B-Instruct")

    if skip:
        print("Skipping model download (--skip-models flag).")
        return

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    gguf_path = models_dir / "qwen3-72b-instruct-q4_k_m.gguf"
    hf_path = models_dir / "qwen3-72b"

    if gguf_path.exists():
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"GGUF model already exists ({size_gb:.1f} GB): {gguf_path}")
        if size_gb < 30:
            print("WARNING: File is smaller than expected — may be incomplete. Re-downloading.")
        else:
            return

    print("Attempting to download GGUF (Q4_K_M) via huggingface-cli...")
    huggingface_cli = VENV_DIR / "bin" / "huggingface-cli"

    # Try GGUF first
    rc = subprocess.run(
        f"{huggingface_cli} download Qwen/Qwen3-72B-Instruct-GGUF "
        f"qwen3-72b-instruct-q4_k_m.gguf "
        f"--local-dir {models_dir}",
        shell=True
    ).returncode

    if rc != 0:
        print("GGUF download failed or model not available. Trying HF format for vLLM...")
        if not hf_path.exists():
            rc2 = subprocess.run(
                f"{huggingface_cli} download Qwen/Qwen3-72B-Instruct "
                f"--local-dir {hf_path}",
                shell=True
            ).returncode
            if rc2 != 0:
                print("WARNING: Qwen3-72B download failed. Check HuggingFace access.")
        else:
            print(f"HF model already exists at: {hf_path}")
    else:
        print(f"GGUF model downloaded to: {gguf_path}")

    # Install llama-cpp-python if GGUF is available
    if gguf_path.exists():
        check_result = subprocess.run(
            [str(VENV_PYTHON), "-c", "import llama_cpp"],
            capture_output=True
        )
        if check_result.returncode != 0:
            print("Installing llama-cpp-python for GGUF inference...")
            run(
                f"{VENV_PIP} install llama-cpp-python "
                f"--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121",
                check=False
            )


# ===========================================================================
# STEP 5: Download DeepSeek-Prover-V1.5-RL
# ===========================================================================

def download_deepseek_prover(skip: bool = False) -> None:
    heading("STEP 5: DeepSeek-Prover-V1.5-RL")

    if skip:
        print("Skipping model download (--skip-models flag).")
        return

    models_dir = PROJECT_ROOT / "models"
    prover_path = models_dir / "deepseek-prover-v1.5-rl"

    if prover_path.exists():
        # Check if it has model weights (not just config files)
        weight_files = list(prover_path.glob("*.bin")) + list(prover_path.glob("*.safetensors"))
        if weight_files:
            total_size = sum(f.stat().st_size for f in weight_files) / (1024**3)
            print(f"DeepSeek-Prover already downloaded ({total_size:.1f} GB of weights).")
            return
        else:
            print("DeepSeek-Prover directory exists but weights not found. Re-downloading...")

    huggingface_cli = VENV_DIR / "bin" / "huggingface-cli"
    print(f"Downloading DeepSeek-Prover-V1.5-RL to {prover_path}...")
    rc = subprocess.run(
        f"{huggingface_cli} download deepseek-ai/DeepSeek-Prover-V1.5-RL "
        f"--local-dir {prover_path}",
        shell=True
    ).returncode

    if rc != 0:
        print("WARNING: DeepSeek-Prover download failed. Cross-model scoring will be unavailable.")
    else:
        print(f"DeepSeek-Prover downloaded to: {prover_path}")


# ===========================================================================
# STEP 6: Build corpus vector store
# ===========================================================================

def build_corpus_vectorstore(force_rebuild: bool = False) -> None:
    heading("STEP 6: Corpus Vector Store")

    corpus_dir = PROJECT_ROOT / "corpus"
    vectorstore_dir = PROJECT_ROOT / "vectorstore"

    # Count corpus files
    pdf_files = list(corpus_dir.glob("**/*.pdf"))
    txt_files = list(corpus_dir.glob("**/*.txt"))
    # Exclude README.txt
    txt_files = [f for f in txt_files if f.name != "README.txt"]
    doc_files = pdf_files + txt_files

    if not doc_files:
        print("No documents found in corpus/. Add PDFs or TXTs before running sessions.")
        print(f"See {corpus_dir}/README.txt for instructions.")
        return

    print(f"Found {len(doc_files)} corpus files ({len(pdf_files)} PDFs, {len(txt_files)} TXTs)")

    # Check if vectorstore is up to date
    timestamp_file = vectorstore_dir / ".last_embed_timestamp.json"
    if not force_rebuild and timestamp_file.exists():
        import json as json_mod
        with open(timestamp_file) as f:
            timestamps = json_mod.load(f)
        new_files = [
            f for f in doc_files
            if f.stat().st_mtime > timestamps.get(str(f), 0.0)
        ]
        if not new_files:
            print("Corpus vector store is up to date. Skipping.")
            return
        print(f"{len(new_files)} new/changed files need embedding.")
    else:
        print("Building corpus vector store from scratch...")

    # Import and run the corpus retriever's build function
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        # Activate venv by adding its site-packages to path
        import importlib
        venv_site = VENV_DIR / "lib"
        for d in venv_site.iterdir():
            if d.is_dir() and d.name.startswith("python"):
                site_packages = d / "site-packages"
                if site_packages.exists() and str(site_packages) not in sys.path:
                    sys.path.insert(0, str(site_packages))

        from tools.corpus_retriever import CorpusRetriever
        retriever = CorpusRetriever(
            corpus_dir=str(corpus_dir),
            vectorstore_dir=str(vectorstore_dir),
        )
        n_added = retriever.build_or_update_index(force_rebuild=force_rebuild)
        print(f"Vector store built: {n_added} chunks added.")
    except Exception as e:
        print(f"WARNING: Corpus vector store build failed: {e}")
        print("You can rebuild manually by running: python -c \"from tools.corpus_retriever import CorpusRetriever; CorpusRetriever().build_or_update_index()\"")


# ===========================================================================
# STEP 7: Write example problem prompt if none exists
# ===========================================================================

def write_example_problem() -> None:
    problem_path = PROJECT_ROOT / "problem_prompt.txt"
    if problem_path.exists():
        return

    example = """PROBLEM STATEMENT
One-layer softmax transformers have been shown to fail on certain compositionality tasks. Existing impossibility proofs rely on communication complexity arguments and require a finite precision assumption on the arithmetic used by the transformer.

KNOWN RESULTS (provide these to the Generator as seed context)
- Any one-layer softmax transformer solving function composition must have hidden dimension at least Omega(n) under sub-linear precision assumptions. [Peng et al. 2024]
- The Match-3 task (identifying token triplets satisfying a relation) cannot be solved by a one-layer transformer with O(log n) precision. [Sanford et al. 2023]
- Standard attention computes pairwise interactions between tokens via the product Q*K^T. [Vaswani et al. 2017]
- Matrix multiplication of 2x2 matrices requires at most 7 scalar multiplications rather than 8. [1969]
- The bilinear complexity of a map is the minimum number of scalar multiplications needed to compute it. This is related to the rank of the corresponding tensor.

OPEN QUESTION
Does the impossibility result for function composition survive when the finite precision assumption is removed? If the impossibility persists under infinite precision, propose a modification to the attention mechanism that provably resolves it, and analyze the computational complexity of the proposed modification.

CONSTRAINTS ON YOUR INVESTIGATION
- Work within the framework of one-layer transformers unless you can prove that additional layers are strictly necessary.
- Any proposed modification must be analyzed for expressivity: prove what class of functions it can compute.
- Any proposed modification must be analyzed for computational complexity: what is the cost relative to standard attention?
- Prioritize formal verifiability: every claim should be expressible as a Lean 4 theorem.
"""

    with open(problem_path, "w") as f:
        f.write(example)
    print(f"Example problem prompt written to: {problem_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOVA Reasoning System — Setup and Installation"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip downloading Qwen3-72B and DeepSeek-Prover (saves time if already downloaded)",
    )
    parser.add_argument(
        "--skip-lean",
        action="store_true",
        help="Skip Lean 4 installation and Mathlib build",
    )
    parser.add_argument(
        "--force-rebuild-corpus",
        action="store_true",
        help="Force rebuild of corpus vector store even if up to date",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  NOVA REASONING SYSTEM — SETUP")
    print("  This script is idempotent. Run it as many times as needed.")
    print("=" * 70)

    start = time.time()

    try:
        install_system_deps()
        setup_python_env()
        setup_lean(skip=args.skip_lean)
        download_qwen3(skip=args.skip_models)
        download_deepseek_prover(skip=args.skip_models)
        build_corpus_vectorstore(force_rebuild=args.force_rebuild_corpus)
        write_example_problem()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed: {e}")
        raise

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  Setup complete in {elapsed:.0f}s.")
    print(f"  Next steps:")
    print(f"    1. Add PDFs/TXTs to: {PROJECT_ROOT}/corpus/")
    print(f"    2. Edit problem prompt: {PROJECT_ROOT}/problem_prompt.txt")
    print(f"    3. Run: python main.py --problem ./problem_prompt.txt")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
