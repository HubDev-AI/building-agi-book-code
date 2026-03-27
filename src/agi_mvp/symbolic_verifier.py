# symbolic_verifier.py

from dataclasses import dataclass
from typing import Any
import z3

from .reasoning_types import Constraint, VerificationStatus


@dataclass
class VerificationResult:
    status: VerificationStatus
    detail: str
    model: dict[str, Any] | None = None


class SymbolicVerifier:
    """Verifies claims and solves constraints using Z3."""

    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms

    def verify(
        self,
        claim_constraints: list[Constraint],
        context_constraints: list[Constraint] | None = None,
    ) -> VerificationResult:
        """Check whether claim constraints are satisfiable
        given optional context constraints."""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        namespace = self._build_namespace()

        # Add context constraints (known truths)
        if context_constraints:
            for c in context_constraints:
                try:
                    expr = eval(c.expression, namespace)
                    solver.add(expr)
                except Exception as e:
                    return VerificationResult(
                        status=VerificationStatus.UNKNOWN,
                        detail=(
                            f"Failed to parse context "
                            f"constraint '{c.name}': {e}"
                        ),
                    )

        # Add claim constraints
        for c in claim_constraints:
            try:
                expr = eval(c.expression, namespace)
                solver.add(expr)
            except Exception as e:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    detail=(
                        f"Failed to parse claim "
                        f"constraint '{c.name}': {e}"
                    ),
                )

        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            assignments = {
                str(d): str(model[d])
                for d in model.decls()
            }
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                detail="All constraints satisfiable.",
                model=assignments,
            )
        elif result == z3.unsat:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                detail=(
                    f"Constraints are unsatisfiable. "
                    f"Core: {solver.unsat_core()}"
                    if solver.unsat_core()
                    else "Constraints are unsatisfiable."
                ),
            )
        else:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                detail="Z3 returned unknown (likely timeout).",
            )

    def solve_constraints(
        self,
        variables: dict[str, str],
        constraints: list[Constraint],
    ) -> VerificationResult:
        """Find variable assignments satisfying all
        constraints.

        Args:
            variables: name -> type mapping. Supported types:
                       "int", "real", "bool".
            constraints: Z3 expressions referencing the
                         variable names.
        """
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        namespace = self._build_namespace()

        # Declare variables
        type_map = {
            "int": z3.Int, "real": z3.Real, "bool": z3.Bool,
        }
        for var_name, var_type in variables.items():
            constructor = type_map.get(var_type)
            if constructor is None:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    detail=(
                        f"Unsupported variable type: "
                        f"{var_type}"
                    ),
                )
            namespace[var_name] = constructor(var_name)

        # Add constraints
        for c in constraints:
            try:
                expr = eval(c.expression, namespace)
                solver.add(expr)
            except Exception as e:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    detail=(
                        f"Failed to parse constraint "
                        f"'{c.name}': {e}"
                    ),
                )

        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            assignments = {}
            for var_name in variables:
                z3_var = namespace[var_name]
                val = model.evaluate(
                    z3_var, model_completion=True
                )
                assignments[var_name] = str(val)
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                detail="Solution found.",
                model=assignments,
            )
        elif result == z3.unsat:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                detail=(
                    "No solution exists -- constraints "
                    "are unsatisfiable."
                ),
            )
        else:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                detail="Solver timed out.",
            )

    @staticmethod
    def _build_namespace() -> dict:
        """Build the evaluation namespace with Z3 primitives
        and common variable names pre-declared."""
        ns: dict = {"z3": z3, "__builtins__": {}}
        for name in [
            "Int", "Real", "Bool", "IntVal", "RealVal",
            "BoolVal", "And", "Or", "Not", "Implies", "If",
            "Distinct", "Sum", "Product", "ForAll", "Exists",
        ]:
            ns[name] = getattr(z3, name)
        # Pre-declare common variable names so constraints
        # like "x > 0" work without explicit declaration
        for v in "abcdefghijklmnopqrstuvwxyz":
            ns[v] = z3.Int(v)
        return ns
