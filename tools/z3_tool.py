"""
tools/z3_tool.py — Z3 SMT solver wrapper for the NOVA Reasoning System.

The Generator can call z3_check(constraints) to verify the satisfiability
of logical constraints arising in proof attempts. This is particularly useful
for:
  - Verifying that a proposed counterexample to an assumed impossibility
    actually satisfies the constraints.
  - Checking that a proposed relaxed assumption is logically weaker than
    the original (i.e., the original implies the relaxation but not vice versa).
  - Verifying dimension-counting and cardinality arguments that underlie
    complexity-theoretic bounds.

The tool accepts constraints in a simple JSON-based DSL that is translated
to Z3 Python API calls. This avoids requiring the Generator to know Z3
syntax directly.

Constraint DSL (subset supported):
  {"type": "implies", "lhs": ..., "rhs": ...}
  {"type": "and", "args": [...]}
  {"type": "or", "args": [...]}
  {"type": "not", "arg": ...}
  {"type": "leq", "lhs": ..., "rhs": ...}  -- <=
  {"type": "geq", "lhs": ..., "rhs": ...}  -- >=
  {"type": "eq",  "lhs": ..., "rhs": ...}  -- ==
  {"type": "lt",  "lhs": ..., "rhs": ...}  -- <
  {"type": "gt",  "lhs": ..., "rhs": ...}  -- >
  {"type": "var", "name": "x", "sort": "Int" | "Real" | "Bool"}
  {"type": "const", "value": 42}
  {"type": "add", "args": [...]}
  {"type": "mul", "args": [...]}
  {"type": "sub", "lhs": ..., "rhs": ...}
"""

import io
import json
import time
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import z3

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Z3Result:
    """Result of a Z3 satisfiability or validity check."""
    query_type: str           # "sat" | "unsat" | "valid" | "invalid" | "unknown" | "error"
    satisfiable: Optional[bool]  # True=SAT, False=UNSAT, None=unknown
    model: Optional[Dict[str, Any]]  # variable assignments if SAT
    unsat_core: Optional[List[str]]  # constraint names in unsat core if UNSAT
    elapsed_seconds: float
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "satisfiable": self.satisfiable,
            "model": self.model,
            "unsat_core": self.unsat_core,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


class Z3Tool:
    """
    Z3 constraint solver wrapper.

    Accepts either:
      1. A JSON-serializable dict describing constraints in the DSL above.
      2. A list of such dicts (all added to the same solver).
      3. A plain Python string containing Z3 Python API code that prints
         SAT / UNSAT / UNKNOWN on stdout.
    """

    def check(
        self,
        constraints: Union[Dict[str, Any], str, List],
        timeout_seconds: int = 30,
        check_validity: bool = False,
    ) -> Z3Result:
        """
        Check satisfiability (or validity if check_validity=True) of constraints.

        Args:
            constraints:     JSON DSL dict, list of DSL dicts, or Python Z3 code string.
            timeout_seconds: Hard timeout in seconds.
            check_validity:  If True, check if constraints are VALID (tautology)
                             by negating them and checking for UNSAT.

        Returns:
            Z3Result with satisfiability verdict and optional model / unsat core.
        """
        start = time.time()
        try:
            if isinstance(constraints, str):
                return self._check_python_code(constraints, timeout_seconds, check_validity)
            elif isinstance(constraints, (dict, list)):
                return self._check_dsl(constraints, timeout_seconds, check_validity)
            else:
                return Z3Result(
                    query_type="error",
                    satisfiable=None,
                    model=None,
                    unsat_core=None,
                    elapsed_seconds=time.time() - start,
                    error=f"Unknown constraint type: {type(constraints).__name__}",
                )
        except Exception as exc:
            log.error(f"Z3 check error: {exc}")
            return Z3Result(
                query_type="error",
                satisfiable=None,
                model=None,
                unsat_core=None,
                elapsed_seconds=time.time() - start,
                error=str(exc),
            )

    def implies(
        self,
        hypothesis: Union[Dict, str],
        conclusion: Union[Dict, str],
        timeout_seconds: int = 30,
    ) -> Z3Result:
        """
        Check whether hypothesis logically implies conclusion.

        Strategy: check SAT(hypothesis AND NOT conclusion).
        If UNSAT: implication is valid.
        If SAT: counterexample exists; implication is invalid.
        """
        if isinstance(hypothesis, dict) and isinstance(conclusion, dict):
            combined = {
                "type": "and",
                "args": [hypothesis, {"type": "not", "arg": conclusion}],
            }
            result = self.check(combined, timeout_seconds=timeout_seconds)
            if result.satisfiable is False:
                return Z3Result(
                    query_type="valid",
                    satisfiable=None,
                    model=None,
                    unsat_core=None,
                    elapsed_seconds=result.elapsed_seconds,
                    error=None,
                )
            elif result.satisfiable is True:
                return Z3Result(
                    query_type="invalid",
                    satisfiable=True,
                    model=result.model,
                    unsat_core=None,
                    elapsed_seconds=result.elapsed_seconds,
                    error=None,
                )
            return result
        else:
            raise ValueError(
                "implies() requires both hypothesis and conclusion as DSL dicts. "
                "For Python code, use check() directly."
            )

    def check_independence(
        self,
        assumption_a: Dict,
        assumption_b: Dict,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Check whether two assumptions are logically independent.

        Returns:
          {
            "a_implies_b": bool | None,
            "b_implies_a": bool | None,
            "independent": bool,
          }
        """
        r_ab = self.implies(assumption_a, assumption_b, timeout_seconds)
        r_ba = self.implies(assumption_b, assumption_a, timeout_seconds)
        a_implies_b = r_ab.query_type == "valid"
        b_implies_a = r_ba.query_type == "valid"
        return {
            "a_implies_b": a_implies_b,
            "b_implies_a": b_implies_a,
            "independent": not a_implies_b and not b_implies_a,
        }

    # -----------------------------------------------------------------------
    # PRIVATE: DSL TRANSLATION
    # -----------------------------------------------------------------------

    def _check_dsl(
        self,
        constraints: Union[Dict, List],
        timeout_seconds: int,
        check_validity: bool,
    ) -> Z3Result:
        """Translate DSL constraints to Z3 expressions and solve."""
        solver = z3.Solver()
        solver.set("timeout", timeout_seconds * 1000)  # Z3 uses milliseconds

        vars_: Dict[str, Any] = {}

        exprs = []
        if isinstance(constraints, list):
            exprs = [self._translate(c, vars_) for c in constraints]
        else:
            exprs = [self._translate(constraints, vars_)]

        if check_validity:
            # validity: check if negation is UNSAT
            for expr in exprs:
                solver.add(z3.Not(expr))
        else:
            for expr in exprs:
                solver.add(expr)

        start = time.time()
        verdict = solver.check()
        elapsed = time.time() - start

        if verdict == z3.sat:
            model = solver.model()
            model_dict: Dict[str, Any] = {}
            for decl in model.decls():
                try:
                    model_dict[decl.name()] = str(model[decl])
                except Exception:
                    pass
            return Z3Result(
                query_type="invalid" if check_validity else "sat",
                satisfiable=True,
                model=model_dict,
                unsat_core=None,
                elapsed_seconds=elapsed,
                error=None,
            )
        elif verdict == z3.unsat:
            return Z3Result(
                query_type="valid" if check_validity else "unsat",
                satisfiable=False,
                model=None,
                unsat_core=None,
                elapsed_seconds=elapsed,
                error=None,
            )
        else:
            return Z3Result(
                query_type="unknown",
                satisfiable=None,
                model=None,
                unsat_core=None,
                elapsed_seconds=elapsed,
                error="z3_returned_unknown",
            )

    def _translate(self, constraint: Any, vars_: Dict) -> Any:
        """Recursively translate a DSL constraint dict to a Z3 expression."""
        if isinstance(constraint, bool):
            return z3.BoolVal(constraint)
        if isinstance(constraint, int):
            return z3.IntVal(constraint)
        if isinstance(constraint, float):
            return z3.RealVal(constraint)
        if not isinstance(constraint, dict):
            raise ValueError(f"Cannot translate constraint of type {type(constraint).__name__}: {constraint!r}")

        t = constraint.get("type", "")

        if t == "var":
            name = constraint["name"]
            sort = constraint.get("sort", "Int")
            if name not in vars_:
                if sort == "Int":
                    vars_[name] = z3.Int(name)
                elif sort == "Real":
                    vars_[name] = z3.Real(name)
                elif sort == "Bool":
                    vars_[name] = z3.Bool(name)
                else:
                    raise ValueError(f"Unknown sort: {sort!r}")
            return vars_[name]

        if t == "const":
            v = constraint["value"]
            if isinstance(v, bool):
                return z3.BoolVal(v)
            if isinstance(v, int):
                return z3.IntVal(v)
            return z3.RealVal(float(v))

        if t == "and":
            return z3.And(*[self._translate(a, vars_) for a in constraint["args"]])

        if t == "or":
            return z3.Or(*[self._translate(a, vars_) for a in constraint["args"]])

        if t == "not":
            return z3.Not(self._translate(constraint["arg"], vars_))

        if t == "implies":
            return z3.Implies(
                self._translate(constraint["lhs"], vars_),
                self._translate(constraint["rhs"], vars_),
            )

        if t == "eq":
            return self._translate(constraint["lhs"], vars_) == self._translate(constraint["rhs"], vars_)

        if t == "leq":
            return self._translate(constraint["lhs"], vars_) <= self._translate(constraint["rhs"], vars_)

        if t == "geq":
            return self._translate(constraint["lhs"], vars_) >= self._translate(constraint["rhs"], vars_)

        if t == "lt":
            return self._translate(constraint["lhs"], vars_) < self._translate(constraint["rhs"], vars_)

        if t == "gt":
            return self._translate(constraint["lhs"], vars_) > self._translate(constraint["rhs"], vars_)

        if t == "add":
            args = [self._translate(a, vars_) for a in constraint["args"]]
            return z3.Sum(args)

        if t == "mul":
            args = [self._translate(a, vars_) for a in constraint["args"]]
            result = args[0]
            for a in args[1:]:
                result = result * a
            return result

        if t == "sub":
            return (
                self._translate(constraint["lhs"], vars_)
                - self._translate(constraint["rhs"], vars_)
            )

        if t == "div":
            return (
                self._translate(constraint["lhs"], vars_)
                / self._translate(constraint["rhs"], vars_)
            )

        if t == "ite":
            return z3.If(
                self._translate(constraint["cond"], vars_),
                self._translate(constraint["then"], vars_),
                self._translate(constraint["else"], vars_),
            )

        raise ValueError(f"Unknown DSL constraint type: {t!r}")

    # -----------------------------------------------------------------------
    # PRIVATE: PYTHON CODE EXECUTION PATH
    # -----------------------------------------------------------------------

    def _check_python_code(
        self,
        code: str,
        timeout_seconds: int,
        check_validity: bool,
    ) -> Z3Result:
        """
        Execute Python code that uses the Z3 API directly.
        The code must print "SAT", "UNSAT", or "UNKNOWN" to stdout.
        Only z3 and standard math operations are permitted in the namespace.
        """
        start = time.time()

        # Construct a minimal namespace with only z3 and safe builtins
        import math
        namespace: Dict[str, Any] = {
            "z3": z3,
            "math": math,
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "True": True,
                "False": False,
                "None": None,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "min": min,
                "max": max,
                "abs": abs,
                "sum": sum,
                "any": any,
                "all": all,
            },
        }

        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code, namespace)  # noqa: S102
        except Exception as exc:
            return Z3Result(
                query_type="error",
                satisfiable=None,
                model=None,
                unsat_core=None,
                elapsed_seconds=time.time() - start,
                error=f"exec_error: {exc}",
            )

        output = output_buffer.getvalue().strip().upper()
        elapsed = time.time() - start

        if "UNSAT" in output:
            qtype = "valid" if check_validity else "unsat"
            return Z3Result(qtype, False, None, None, elapsed, None)
        elif "SAT" in output:
            qtype = "invalid" if check_validity else "sat"
            return Z3Result(qtype, True, None, None, elapsed, None)
        else:
            return Z3Result("unknown", None, None, None, elapsed, None)
