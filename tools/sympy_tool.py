"""
tools/sympy_tool.py — SymPy symbolic algebra wrapper for the NOVA Reasoning System.

The Generator can call sympy_compute(expression) to perform symbolic algebra
operations that are too complex for manual computation but too simple for Lean.

Use cases in this system:
  - Computing ranks of specific small matrices and tensors.
  - Verifying that a claimed algebraic complexity bound is numerically achievable.
  - Performing dimensional analysis on linear maps and operators.
  - Computing characteristic polynomials and eigenvalue structure.
  - Verifying algebraic identities that arise in proof sketches.
  - Simplifying expressions in dimension-counting arguments.

The tool accepts either:
  1. A Python expression string using SymPy notation, evaluated in a namespace
     that has SymPy symbols pre-imported.
  2. A structured command dict for common operations (rank, simplify, etc.).

Returns a SymPyResult with the computed value and its LaTeX representation.
"""

import io
import contextlib
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import sympy as sp
from sympy import (
    Matrix, symbols, Symbol, Integer, Rational, sqrt, pi, exp, log,
    simplify, expand, factor, collect, apart, cancel, nsimplify,
    solve, groebner, resultant, det, trace, eigenvals, eigenvects,
    tensorproduct, KroneckerDelta, MatrixSymbol, BlockMatrix,
    latex, pretty, srepr,
)
from sympy.tensor.array import Array, tensorproduct as atensorproduct
from sympy.abc import x, y, z, n, m, k

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SymPyResult:
    """Result of a SymPy computation."""
    success: bool
    result_str: str          # str() of the result
    result_latex: str        # LaTeX of the result
    result_numerical: Optional[float]   # float approximation if applicable
    computation_type: str    # what was computed
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result_str": self.result_str,
            "result_latex": self.result_latex,
            "result_numerical": self.result_numerical,
            "computation_type": self.computation_type,
            "error": self.error,
        }


class SymPyTool:
    """SymPy symbolic computation wrapper."""

    # Namespace available to all eval'd expressions
    _SYMPY_NAMESPACE: Dict[str, Any] = {
        "sp": sp,
        "Matrix": Matrix,
        "symbols": symbols,
        "Symbol": Symbol,
        "Integer": Integer,
        "Rational": Rational,
        "sqrt": sqrt,
        "pi": pi,
        "exp": exp,
        "log": log,
        "simplify": simplify,
        "expand": expand,
        "factor": factor,
        "collect": collect,
        "apart": apart,
        "cancel": cancel,
        "nsimplify": nsimplify,
        "solve": solve,
        "groebner": groebner,
        "resultant": resultant,
        "det": det,
        "trace": trace,
        "eigenvals": eigenvals,
        "eigenvects": eigenvects,
        "latex": latex,
        "pretty": pretty,
        "Array": Array,
        "KroneckerDelta": KroneckerDelta,
        "MatrixSymbol": MatrixSymbol,
        "BlockMatrix": BlockMatrix,
        "tensorproduct": atensorproduct,
        # Common symbols pre-bound for convenience
        "x": x, "y": y, "z": z, "n": n, "m": m, "k": k,
        # Math builtins
        "range": range,
        "len": len,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "print": print,
        "str": str,
        "int": int,
        "float": float,
        "True": True,
        "False": False,
        "None": None,
        "isinstance": isinstance,
        "enumerate": enumerate,
        "zip": zip,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
    }

    def compute(
        self,
        expression: Union[str, Dict[str, Any]],
    ) -> SymPyResult:
        """
        Evaluate a SymPy expression or command.

        Args:
            expression: Either a Python expression string (SymPy namespace available)
                        or a structured command dict (see below).

        Command dict format:
          {"op": "rank",      "matrix": [[1,2],[3,4]]}
          {"op": "det",       "matrix": [[1,2],[3,4]]}
          {"op": "eigenvals", "matrix": [[1,2],[3,4]]}
          {"op": "simplify",  "expr": "x**2 + 2*x + 1"}
          {"op": "factor",    "expr": "x**3 - 1"}
          {"op": "solve",     "expr": "x**2 - 2", "vars": ["x"]}
          {"op": "tensor_rank_upper_bound", "shape": [2,2,2]}
          {"op": "analyze_multilinear_map", "tensors": [...], "field": "QQ"}

        Returns:
            SymPyResult with the computed value.
        """
        if isinstance(expression, dict):
            return self._command(expression)
        elif isinstance(expression, str):
            return self._eval_expression(expression)
        else:
            return SymPyResult(
                success=False,
                result_str="",
                result_latex="",
                result_numerical=None,
                computation_type="unknown",
                error=f"Unsupported expression type: {type(expression).__name__}",
            )

    def matrix_rank(self, matrix_data: List[List]) -> SymPyResult:
        """Compute the rank of a matrix."""
        try:
            M = Matrix(matrix_data)
            rank = M.rank()
            return SymPyResult(
                success=True,
                result_str=str(rank),
                result_latex=latex(rank),
                result_numerical=float(rank),
                computation_type="matrix_rank",
                error=None,
            )
        except Exception as e:
            return self._error_result("matrix_rank", e)

    def analyze_multilinear_map(
        self,
        tensors: List[List],
        field: str = "QQ",
    ) -> SymPyResult:
        """
        Analyze the rank structure of a multilinear map given as a list of tensors.

        Each tensor in `tensors` is provided as a nested list (matrix or higher-order
        array). The function computes the rank of each matrix slice and returns
        a summary of rank bounds.

        Args:
            tensors: List of matrices (as nested lists) representing the map.
            field:   Field over which to compute ranks ("QQ" or "RR").

        Returns:
            SymPyResult with rank information for each input tensor.
        """
        try:
            results = []
            total_rank = 0
            for i, t in enumerate(tensors):
                M = Matrix(t)
                r = M.rank()
                total_rank += int(r)
                nrows, ncols = M.shape
                results.append(
                    f"Tensor {i}: shape {nrows}x{ncols}, rank={r}"
                )

            summary = (
                f"Multilinear map analysis ({field}):\n"
                + "\n".join(f"  {line}" for line in results)
                + f"\n  Combined rank sum: {total_rank}"
            )

            return SymPyResult(
                success=True,
                result_str=summary,
                result_latex=r"\text{rank sum} = " + str(total_rank),
                result_numerical=float(total_rank),
                computation_type="analyze_multilinear_map",
                error=None,
            )
        except Exception as e:
            return self._error_result("analyze_multilinear_map", e)

    def tensor_contraction_complexity(
        self,
        shape_a: List[int],
        shape_b: List[int],
        contraction_axes: List[int],
    ) -> SymPyResult:
        """
        Compute the standard arithmetic complexity of a tensor contraction.
        Useful for analyzing the complexity of generalized multilinear operations.
        """
        try:
            # Multiplications = product of all dimensions
            a_syms = [Symbol(f"n_{i}") for i in range(len(shape_a))]
            b_syms = [Symbol(f"m_{i}") for i in range(len(shape_b))]

            # Dimensions not contracted
            free_a = [d for i, d in enumerate(shape_a) if i not in contraction_axes]
            free_b = [d for i, d in enumerate(shape_b) if i not in contraction_axes]
            contracted = [shape_a[i] for i in contraction_axes]

            output_dims = free_a + free_b
            output_size = 1
            for d in output_dims:
                output_size *= d
            contract_size = 1
            for d in contracted:
                contract_size *= d

            total_mults = output_size * contract_size
            result = sp.Integer(total_mults)

            return SymPyResult(
                success=True,
                result_str=f"Contraction complexity: {total_mults} multiplications\n"
                           f"Output shape: {output_dims}\n"
                           f"Contracted dimensions: {contracted}",
                result_latex=latex(result),
                result_numerical=float(total_mults),
                computation_type="tensor_contraction_complexity",
                error=None,
            )
        except Exception as e:
            return self._error_result("tensor_contraction_complexity", e)

    def compute_bilinear_map_complexity(
        self,
        input_dims: List[int],
        output_dim: int,
    ) -> SymPyResult:
        """
        Compute the arithmetic complexity of an arbitrary bilinear map.

        A bilinear map B: V_1 x V_2 -> W where dim(V_i) = input_dims[i]
        and dim(W) = output_dim has naive complexity
          input_dims[0] * input_dims[1] * output_dim
        scalar multiplications (one per output-input pair).

        Args:
            input_dims:  List of two integers [dim_V1, dim_V2].
            output_dim:  Dimension of the output space.

        Returns:
            SymPyResult with the naive complexity and symbolic expression.
        """
        try:
            if len(input_dims) != 2:
                raise ValueError(
                    f"Bilinear map requires exactly 2 input dimensions, got {len(input_dims)}"
                )
            d1, d2 = input_dims
            out = output_dim

            d1_sym, d2_sym, out_sym = symbols('d1 d2 out', positive=True, integer=True)

            naive_cost_sym = d1_sym * d2_sym * out_sym
            naive_cost_val = d1 * d2 * out

            result_str = (
                f"Bilinear map complexity:\n"
                f"  Input dimensions: {d1} x {d2}\n"
                f"  Output dimension: {out}\n"
                f"  Naive cost (multiplications): {d1} * {d2} * {out} = {naive_cost_val}\n"
                f"  Symbolic: {naive_cost_sym}"
            )

            return SymPyResult(
                success=True,
                result_str=result_str,
                result_latex=latex(naive_cost_sym),
                result_numerical=float(naive_cost_val),
                computation_type="compute_bilinear_map_complexity",
                error=None,
            )
        except Exception as e:
            return self._error_result("compute_bilinear_map_complexity", e)

    # -----------------------------------------------------------------------
    # PRIVATE: EXPRESSION EVAL
    # -----------------------------------------------------------------------

    def _eval_expression(self, expression: str) -> SymPyResult:
        """Evaluate a SymPy expression string."""
        output_buffer = io.StringIO()
        ns = dict(self._SYMPY_NAMESPACE)

        try:
            with contextlib.redirect_stdout(output_buffer):
                result = eval(expression, ns)  # noqa: S307

            printed = output_buffer.getvalue().strip()
            result_str = str(result) if result is not None else printed
            result_latex_str = ""
            result_num = None

            try:
                result_latex_str = latex(result)
            except Exception:
                result_latex_str = result_str

            try:
                result_num = float(result.evalf() if hasattr(result, 'evalf') else result)
            except Exception:
                pass

            return SymPyResult(
                success=True,
                result_str=result_str,
                result_latex=result_latex_str,
                result_numerical=result_num,
                computation_type="eval",
                error=None,
            )
        except SyntaxError:
            # Try exec for multi-line expressions
            return self._exec_expression(expression)
        except Exception as e:
            return self._error_result("eval", e)

    def _exec_expression(self, expression: str) -> SymPyResult:
        """Execute a multi-line SymPy code block."""
        output_buffer = io.StringIO()
        ns = dict(self._SYMPY_NAMESPACE)
        ns["__result__"] = None

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(expression, ns)  # noqa: S102

            output = output_buffer.getvalue().strip()
            # If code set __result__, use that; otherwise use stdout
            result_val = ns.get("__result__")
            result_str = str(result_val) if result_val is not None else output

            result_latex_str = ""
            try:
                if result_val is not None:
                    result_latex_str = latex(result_val)
            except Exception:
                result_latex_str = result_str

            return SymPyResult(
                success=True,
                result_str=result_str,
                result_latex=result_latex_str,
                result_numerical=None,
                computation_type="exec",
                error=None,
            )
        except Exception as e:
            return self._error_result("exec", e)

    def _command(self, cmd: Dict[str, Any]) -> SymPyResult:
        """Handle structured command dicts."""
        op = cmd.get("op", "")

        if op == "rank":
            return self.matrix_rank(cmd["matrix"])

        if op == "det":
            try:
                M = Matrix(cmd["matrix"])
                d = M.det()
                return SymPyResult(True, str(d), latex(d), float(d.evalf()), "det", None)
            except Exception as e:
                return self._error_result("det", e)

        if op == "eigenvals":
            try:
                M = Matrix(cmd["matrix"])
                evs = M.eigenvals()
                s = str(evs)
                return SymPyResult(True, s, latex(evs), None, "eigenvals", None)
            except Exception as e:
                return self._error_result("eigenvals", e)

        if op == "simplify":
            try:
                expr = sp.sympify(cmd["expr"])
                s = simplify(expr)
                return SymPyResult(True, str(s), latex(s), None, "simplify", None)
            except Exception as e:
                return self._error_result("simplify", e)

        if op == "factor":
            try:
                expr = sp.sympify(cmd["expr"])
                f_result = factor(expr)
                return SymPyResult(True, str(f_result), latex(f_result), None, "factor", None)
            except Exception as e:
                return self._error_result("factor", e)

        if op == "solve":
            try:
                expr = sp.sympify(cmd["expr"])
                var_list = [sp.Symbol(v) for v in cmd.get("vars", ["x"])]
                sol = solve(expr, var_list)
                return SymPyResult(True, str(sol), latex(sol), None, "solve", None)
            except Exception as e:
                return self._error_result("solve", e)

        if op == "analyze_multilinear_map":
            return self.analyze_multilinear_map(
                tensors=cmd.get("tensors", []),
                field=cmd.get("field", "QQ"),
            )

        if op == "compute_bilinear_map_complexity":
            return self.compute_bilinear_map_complexity(
                input_dims=cmd.get("input_dims", [64, 64]),
                output_dim=cmd.get("output_dim", 64),
            )

        return SymPyResult(
            success=False,
            result_str="",
            result_latex="",
            result_numerical=None,
            computation_type=op,
            error=f"Unknown operation: {op!r}",
        )

    @staticmethod
    def _error_result(computation_type: str, exc: Exception) -> SymPyResult:
        log.warning(f"SymPy {computation_type} error: {exc}")
        return SymPyResult(
            success=False,
            result_str="",
            result_latex="",
            result_numerical=None,
            computation_type=computation_type,
            error=f"{type(exc).__name__}: {exc}",
        )
