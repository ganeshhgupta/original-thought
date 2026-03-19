"""
agents/generator.py — Generator Agent for the NOVA Reasoning System.

The Generator is the primary reasoning agent. It runs for GENERATOR_FREE_STEPS
uninterrupted before any critic engages. This replicates the human ability to
follow a chain of reasoning through temporarily unintuitive territory before
evaluating whether it leads somewhere valid.

The Generator has access to:
  - scratchpad_write/read/list (structured working memory)
  - lean_verify (formal verification)
  - z3_check (constraint satisfiability)
  - sympy_compute (symbolic algebra)
  - corpus_retrieve (semantic search over paper corpus)

Tool calls are parsed from the Generator's output as structured JSON blocks
embedded within its reasoning text. This allows the Generator to interleave
free-form reasoning with formal verification steps.

IMPORTANT: The Generator is told it is working on an open problem. It does
not know that a correct answer exists or what it might be.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from memory.scratchpad import Scratchpad
from tools.lean_tool import LeanTool, LeanResult
from tools.z3_tool import Z3Tool
from tools.sympy_tool import SymPyTool
from tools.corpus_retriever import CorpusRetriever
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Generator system prompt
# CRITICAL: No reference to the target result anywhere in this prompt.
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = """You are a mathematical reasoning agent working on an open problem in mathematics.
Your goal is to investigate whether a known limitation in the literature can be overcome through a principled mathematical construction.
You have access to the following tools:

scratchpad_write(name, content, type): write a named result to the scratchpad (type: axiom/established/assumption/conjecture)
scratchpad_read(name): read a named result from the scratchpad
scratchpad_list(): list all current scratchpad entries by type
lean_verify(statement, proof_sketch): attempt to verify a claim in Lean 4
z3_check(constraints): check satisfiability of logical constraints
sympy_compute(expression): compute symbolic algebra
corpus_retrieve(query, top_k): retrieve relevant passages from the paper corpus

CRITICAL RULES:
1. Before making any claim that depends on a prior result, write that result to the scratchpad first.
2. Every proof sketch must be structured as: AXIOMS | ESTABLISHED | ASSUMPTIONS | PROOF STEPS
3. You must call lean_verify on any claim you are confident about before proceeding.
4. You may run for multiple reasoning steps before critics engage. Use this freedom.
5. Follow chains of reasoning through unintuitive territory. Evaluate at the end, not during.
6. When you are stuck, call corpus_retrieve with a structural description, not a topic name.
7. If you notice that two results in your scratchpad seem to be in tension, explicitly name the tension. Do not resolve it prematurely.
8. Do not reference a result without first writing it to the scratchpad.

TOOL CALL FORMAT:
To call a tool, include a JSON block in your response wrapped in <tool_call> tags:
<tool_call>
{"tool": "scratchpad_write", "args": {"name": "lemma1", "content": "For all n >= 1, ...", "type": "assumption"}}
</tool_call>

The tool result will be provided, then you continue reasoning.

PROOF SKETCH FORMAT:
Every proof sketch you write must include these sections:
AXIOMS: (list the axioms being assumed)
ESTABLISHED: (list previously verified results being used)
ASSUMPTIONS: (list unverified claims being relied upon)
PROOF STEPS:
  1. [step]
  2. [step]
  ...
"""

GENERATOR_FRAMING_INJECTIONS = {
    "limitation": (
        "\nFRAMING: Emphasize what is known to fail and precisely why. "
        "The known impossibility results are your starting point. "
        "Work backwards from the failure: what specific property of the architecture causes it? "
        "Is that property essential or accidental?"
    ),
    "technique": (
        "\nFRAMING: Focus on identifying which proof technique family might overcome the current obstacle. "
        "What class of mathematical tools has been used on structurally similar problems? "
        "What is the closest known technique that partially applies here?"
    ),
    "analogy": (
        "\nFRAMING: Look for structural parallels to problems in different mathematical domains. "
        "The current problem has an algebraic structure. What other problems share that structure? "
        "How were they solved?"
    ),
    "assumption": (
        "\nFRAMING: Identify which assumption in the existing impossibility proofs is doing the most work. "
        "What if that assumption were removed or weakened? "
        "Does the impossibility survive under weaker assumptions?"
    ),
    "compression": (
        "\nFRAMING: Look for a single unified explanation for two known results that currently seem separate. "
        "What theorem would simultaneously explain both the known impossibility AND a known positive result?"
    ),
}


class GeneratorOutput:
    """Structured output from a Generator run."""

    def __init__(self) -> None:
        self.reasoning_steps: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_results: List[Dict[str, Any]] = []
        self.scratchpad_writes: List[Dict] = []
        self.lean_calls: List[Dict] = []
        self.z3_calls: List[Dict] = []
        self.sympy_calls: List[Dict] = []
        self.corpus_queries: List[Dict] = []
        self.final_statement: str = ""
        self.raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "scratchpad_writes": self.scratchpad_writes,
            "lean_calls": self.lean_calls,
            "z3_calls": self.z3_calls,
            "sympy_calls": self.sympy_calls,
            "corpus_queries": self.corpus_queries,
            "final_statement": self.final_statement,
        }


class GeneratorAgent:
    """
    Generator Agent: primary mathematical reasoner.

    Runs GENERATOR_FREE_STEPS reasoning iterations before critics engage.
    Each step may include tool calls that update the scratchpad and invoke
    formal verification tools.
    """

    def __init__(
        self,
        lean_tool: LeanTool,
        z3_tool: Z3Tool,
        sympy_tool: SymPyTool,
        corpus_retriever: CorpusRetriever,
    ) -> None:
        self.lean_tool = lean_tool
        self.z3_tool = z3_tool
        self.sympy_tool = sympy_tool
        self.corpus_retriever = corpus_retriever

    def run(
        self,
        context: str,
        scratchpad: Scratchpad,
        framing: str = "limitation",
        injected_failures: Optional[str] = None,
        injected_analogy: Optional[str] = None,
        free_steps: int = config.GENERATOR_FREE_STEPS,
        adversarial_feedback: Optional[str] = None,
        round_num: int = 1,
    ) -> GeneratorOutput:
        """
        Run the Generator for up to free_steps reasoning iterations.

        Args:
            context:              The current problem context (problem prompt +
                                  round summary).
            scratchpad:           Current working memory state.
            framing:              Framing variant key.
            injected_failures:    Past failure summaries to prepend to context.
            injected_analogy:     Analogy from the Analogy Agent to inject.
            free_steps:           Max reasoning steps before stopping.
            adversarial_feedback: If resuming after redirect, feedback to inject.
            round_num:            Current round number.

        Returns:
            GeneratorOutput with all reasoning steps and tool call results.
        """
        from utils.model_loader import get_qwen3
        qwen3 = get_qwen3()

        output = GeneratorOutput()
        conversation: List[Dict[str, str]] = []

        # Build system prompt with framing injection
        system = GENERATOR_SYSTEM_PROMPT
        framing_inj = GENERATOR_FRAMING_INJECTIONS.get(framing, "")
        if framing_inj:
            system += framing_inj

        conversation.append({"role": "system", "content": system})

        # Build initial user message
        initial_message = self._build_initial_message(
            context=context,
            scratchpad=scratchpad,
            injected_failures=injected_failures,
            injected_analogy=injected_analogy,
            adversarial_feedback=adversarial_feedback,
            round_num=round_num,
        )
        conversation.append({"role": "user", "content": initial_message})

        log.info(f"Generator starting round {round_num} (framing={framing}, free_steps={free_steps})")

        # Main generation loop
        for step in range(free_steps):
            log.debug(f"  Generator step {step + 1}/{free_steps}")

            response = qwen3.generate(
                conversation,
                max_tokens=config.GENERATOR_MAX_TOKENS,
                temperature=config.GENERATOR_TEMPERATURE,
            )

            output.reasoning_steps.append(response)
            output.raw_output += response + "\n\n"

            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                tool_results_text = []
                for tc in tool_calls:
                    result = self._execute_tool_call(tc, scratchpad, output)
                    result_str = json.dumps(result, default=str, indent=2)
                    tool_results_text.append(
                        f"<tool_result>\n{json.dumps({'tool': tc.get('tool'), 'result': result}, default=str, indent=2)}\n</tool_result>"
                    )
                    output.tool_calls.append(tc)
                    output.tool_results.append(result)

                # Add assistant response and tool results to conversation
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": "\n".join(tool_results_text) + "\n\nContinue your reasoning."
                })
            else:
                # No tool calls — continue conversation
                conversation.append({"role": "assistant", "content": response})
                # If this looks like a terminal statement, we can stop early
                if self._is_terminal_statement(response):
                    log.debug(f"  Generator reached terminal statement at step {step + 1}")
                    break
                if step < free_steps - 1:
                    conversation.append({
                        "role": "user",
                        "content": "Continue. What is the next step in your reasoning? Have you checked all your assumptions against the scratchpad?"
                    })

        # Extract final statement
        output.final_statement = self._extract_final_statement(output.reasoning_steps)

        log.info(
            f"Generator completed round {round_num}: "
            f"{len(output.reasoning_steps)} steps, "
            f"{len(output.lean_calls)} Lean calls, "
            f"{len(output.scratchpad_writes)} scratchpad writes"
        )

        return output

    # -----------------------------------------------------------------------
    # PRIVATE: TOOL EXECUTION
    # -----------------------------------------------------------------------

    def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        scratchpad: Scratchpad,
        output: GeneratorOutput,
    ) -> Any:
        """Execute a single tool call and return the result."""
        tool = tool_call.get("tool", "")
        args = tool_call.get("args", {})

        try:
            if tool == "scratchpad_write":
                name = args.get("name", f"entry_{len(output.scratchpad_writes)}")
                content = args.get("content", "")
                entry_type = args.get("type", "assumption")
                scratchpad.write(name, content, entry_type)
                output.scratchpad_writes.append({"name": name, "type": entry_type})
                return {"status": "ok", "name": name, "type": entry_type}

            elif tool == "scratchpad_read":
                name = args.get("name", "")
                entry = scratchpad.read(name)
                return {"status": "ok", "entry": entry}

            elif tool == "scratchpad_list":
                return {"status": "ok", "scratchpad": scratchpad.list_all()}

            elif tool == "lean_verify":
                statement = args.get("statement", "")
                proof_sketch = args.get("proof_sketch", "")
                result = self.lean_tool.verify(statement, proof_sketch)
                output.lean_calls.append({
                    "statement": statement,
                    "success": result.success,
                    "obligations_discharged": result.obligations_discharged,
                })
                return result.to_dict()

            elif tool == "z3_check":
                constraints = args.get("constraints", {})
                check_validity = args.get("check_validity", False)
                result = self.z3_tool.check(constraints, check_validity=check_validity)
                output.z3_calls.append({"query_type": result.query_type})
                return result.to_dict()

            elif tool == "sympy_compute":
                expression = args.get("expression", args.get("expr", ""))
                result = self.sympy_tool.compute(expression)
                output.sympy_calls.append({"computation_type": result.computation_type})
                return result.to_dict()

            elif tool == "corpus_retrieve":
                query = args.get("query", "")
                top_k = int(args.get("top_k", 5))
                chunks = self.corpus_retriever.retrieve(query, top_k=top_k)
                output.corpus_queries.append({"query": query, "results": len(chunks)})
                return {
                    "status": "ok",
                    "results": [c.to_dict() for c in chunks],
                }

            else:
                log.warning(f"Generator called unknown tool: {tool!r}")
                return {"status": "error", "error": f"Unknown tool: {tool!r}"}

        except Exception as e:
            log.error(f"Tool call {tool!r} failed: {e}")
            return {"status": "error", "tool": tool, "error": str(e)}

    # -----------------------------------------------------------------------
    # PRIVATE: PROMPT CONSTRUCTION
    # -----------------------------------------------------------------------

    def _build_initial_message(
        self,
        context: str,
        scratchpad: Scratchpad,
        injected_failures: Optional[str],
        injected_analogy: Optional[str],
        adversarial_feedback: Optional[str],
        round_num: int,
    ) -> str:
        parts = []

        parts.append(f"=== ROUND {round_num} ===\n")
        parts.append(f"PROBLEM CONTEXT:\n{context}\n")

        if injected_failures:
            parts.append(
                f"\nPAST FAILURES FROM PREVIOUS SESSIONS "
                f"(do not repeat these approaches without specifically addressing the obstruction):\n"
                f"{injected_failures}\n"
            )

        # Always inject current scratchpad state
        parts.append(f"\n{scratchpad.context_summary()}\n")

        if injected_analogy:
            parts.append(f"\nANALOGY FROM RELATED DOMAIN:\n{injected_analogy}\n")

        if adversarial_feedback:
            parts.append(
                f"\nFEEDBACK TO ADDRESS:\n{adversarial_feedback}\n"
                f"Consider this feedback carefully. Either address it directly or "
                f"explain why it does not apply.\n"
            )

        parts.append(
            "\nBegin your reasoning. Use tool calls to write to the scratchpad, "
            "run Lean verification, check Z3 constraints, compute with SymPy, "
            "or retrieve from the corpus. "
            "Structure every proof sketch as: AXIOMS | ESTABLISHED | ASSUMPTIONS | PROOF STEPS"
        )

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # PRIVATE: PARSING
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
        """
        Parse <tool_call>...</tool_call> blocks from Generator output.
        Returns list of parsed tool call dicts.
        """
        tool_calls = []
        pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)

        for match in pattern.finditer(text):
            raw_json = match.group(1).strip()
            try:
                tc = json.loads(raw_json)
                if isinstance(tc, dict) and "tool" in tc:
                    tool_calls.append(tc)
            except json.JSONDecodeError:
                # Try to extract tool name and args from malformed JSON
                try:
                    # Common failure: trailing commas
                    cleaned = re.sub(r',\s*([}\]])', r'\1', raw_json)
                    tc = json.loads(cleaned)
                    if isinstance(tc, dict) and "tool" in tc:
                        tool_calls.append(tc)
                except json.JSONDecodeError:
                    log.warning(f"Could not parse tool call: {raw_json[:100]}")

        return tool_calls

    @staticmethod
    def _is_terminal_statement(text: str) -> bool:
        """
        Heuristic: does this reasoning step look like a conclusion?
        Avoids early termination on intermediate steps.
        """
        terminal_indicators = [
            "therefore",
            "in conclusion",
            "we have shown",
            "this proves",
            "the conjecture is",
            "we conclude",
            "QED",
            "proof complete",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in terminal_indicators)

    @staticmethod
    def _extract_final_statement(steps: List[str]) -> str:
        """Extract the most important final statement from the reasoning steps."""
        if not steps:
            return ""
        # Use the last step as the final statement
        last = steps[-1]
        # Try to extract the most conjecture-like sentence
        conjecture_patterns = [
            r'CONJECTURE[:\s]+(.+?)(?:\n|$)',
            r'we conjecture that[:\s]+(.+?)(?:\n|$)',
            r'therefore[,\s]+(.+?)(?:\n|$)',
            r'in conclusion[,\s]+(.+?)(?:\n|$)',
        ]
        for pattern in conjecture_patterns:
            match = re.search(pattern, last, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return last non-empty paragraph
        paragraphs = [p.strip() for p in last.split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else last[:500]
