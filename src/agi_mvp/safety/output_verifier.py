# safety/output_verifier.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Re-uses the symbolic verifier from Chapter 7
from ..reasoning_types import Constraint, VerificationStatus
from ..symbolic_verifier import SymbolicVerifier


class SafetyLevel(Enum):
    INFORMATIONAL = "informational"  # Low risk
    OPERATIONAL = "operational"      # Medium risk
    CONSEQUENTIAL = "consequential"  # High risk
    CRITICAL = "critical"            # Highest risk


@dataclass
class SafetyConstraint:
    """A constraint that must hold for an action to be safe."""
    name: str
    description: str
    z3_expression: str
    level: SafetyLevel
    hard: bool = True  # Hard = block; soft = warn


@dataclass
class SafetyVerdict:
    approved: bool
    level: SafetyLevel
    constraints_checked: int
    constraints_passed: int
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_human_approval: bool = False


class OutputVerifier:
    """Verifies that system outputs satisfy safety
    constraints before they reach the outside world."""

    def __init__(self):
        self.verifier = SymbolicVerifier(timeout_ms=3000)
        self._constraints: dict[
            SafetyLevel, list[SafetyConstraint]
        ] = {level: [] for level in SafetyLevel}
        self._register_default_constraints()

    def _register_default_constraints(self):
        """Register baseline safety constraints.

        These are the minimum constraints that every AGI MVP
        should enforce. Extend this method for domain-specific
        safety requirements.
        """
        self.add_constraint(SafetyConstraint(
            name="no_unbounded_loops",
            description=(
                "Generated code must not contain "
                "unbounded loops"
            ),
            z3_expression="loop_bound >= 0",
            level=SafetyLevel.CONSEQUENTIAL,
        ))
        self.add_constraint(SafetyConstraint(
            name="no_network_without_approval",
            description=(
                "No outbound network calls without "
                "explicit approval"
            ),
            z3_expression="network_calls == 0",
            level=SafetyLevel.CRITICAL,
        ))
        self.add_constraint(SafetyConstraint(
            name="resource_bounds",
            description=(
                "Resource usage within defined limits"
            ),
            z3_expression="memory_mb <= 1024",
            level=SafetyLevel.OPERATIONAL,
        ))

    def add_constraint(self, constraint: SafetyConstraint):
        self._constraints[constraint.level].append(
            constraint
        )

    def verify_action(
        self,
        action: dict,
        level: SafetyLevel,
        context: Optional[dict] = None,
    ) -> SafetyVerdict:
        """Verify that a proposed action satisfies all safety
        constraints at or below the given safety level."""
        applicable = []
        for check_level in SafetyLevel:
            applicable.extend(
                self._constraints[check_level]
            )
            if check_level == level:
                break

        if not applicable:
            return SafetyVerdict(
                approved=True, level=level,
                constraints_checked=0,
                constraints_passed=0,
            )

        violations = []
        warnings = []
        passed = 0

        for constraint in applicable:
            z3_constraints = [Constraint(
                name=constraint.name,
                expression=constraint.z3_expression,
                description=constraint.description,
            )]

            result = self.verifier.verify(z3_constraints)

            if result.status == VerificationStatus.VERIFIED:
                passed += 1
            elif result.status == VerificationStatus.REFUTED:
                if constraint.hard:
                    violations.append(
                        f"[BLOCKED] {constraint.name}: "
                        f"{constraint.description}"
                    )
                else:
                    warnings.append(
                        f"[WARNING] {constraint.name}: "
                        f"{constraint.description}"
                    )
            else:
                # Unknown (timeout)
                if constraint.hard:
                    violations.append(
                        f"[UNVERIFIED] {constraint.name}: "
                        f"could not verify within timeout"
                    )
                else:
                    warnings.append(
                        f"[TIMEOUT] {constraint.name}: "
                        f"verification timed out"
                    )

        requires_human = (
            level in (SafetyLevel.CRITICAL,)
            and len(violations) == 0
        )

        return SafetyVerdict(
            approved=len(violations) == 0,
            level=level,
            constraints_checked=len(applicable),
            constraints_passed=passed,
            violations=violations,
            warnings=warnings,
            requires_human_approval=requires_human,
        )
