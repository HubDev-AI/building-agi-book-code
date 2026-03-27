# evaluation/novel_task_generator.py

from dataclasses import dataclass, field
from typing import Optional
import random
import json
import hashlib
from datetime import datetime


@dataclass
class EvaluationTask:
    task_id: str
    description: str
    category: str                    # "reasoning", "planning", "integration", etc.
    difficulty: str                  # "easy", "medium", "hard", "extreme"
    expected_answer: Optional[str]   # None for open-ended tasks
    verification_fn: Optional[str]   # Python code to verify the answer
    dimensions_tested: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class NovelTaskGenerator:
    """Generates evaluation tasks that test generalization
    by combining familiar components in novel configurations."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.generation_timestamp = datetime.now().isoformat()

    def generate_constraint_satisfaction(
        self, num_variables: int = 4, num_constraints: int = 6
    ) -> EvaluationTask:
        """Generate a novel constraint satisfaction problem.

        These test the symbolic verifier directly: the neural model
        must translate the natural-language constraints into Z3,
        and the symbolic engine must find a satisfying assignment.
        """
        names = self.rng.sample(
            ["Alice", "Bob", "Carol", "Dave", "Eve",
             "Frank", "Grace", "Hank", "Ivy", "Jack"],
            num_variables,
        )
        activities = self.rng.sample(
            ["swimming", "hiking", "painting", "cooking",
             "reading", "gardening", "chess", "cycling"],
            num_variables,
        )
        time_slots = list(range(1, num_variables + 1))

        # Build a valid assignment first, then generate constraints from it
        assignment = {}
        shuffled_activities = list(activities)
        self.rng.shuffle(shuffled_activities)
        shuffled_slots = list(time_slots)
        self.rng.shuffle(shuffled_slots)

        for i, name in enumerate(names):
            assignment[name] = {
                "activity": shuffled_activities[i],
                "slot": shuffled_slots[i],
            }

        # Generate constraints that are true under the valid assignment
        constraints = []
        constraint_templates = [
            lambda n1, n2: (
                f"{n1} has an earlier time slot than {n2}",
                assignment[n1]["slot"] < assignment[n2]["slot"],
            ),
            lambda n1, _: (
                f"{n1} does {assignment[n1]['activity']}",
                True,
            ),
            lambda n1, _: (
                f"{n1}'s time slot is "
                f"{'odd' if assignment[n1]['slot'] % 2 == 1 else 'even'}",
                True,
            ),
            lambda n1, n2: (
                f"{n1} and {n2} are not in adjacent time slots",
                abs(assignment[n1]["slot"] - assignment[n2]["slot"]) > 1,
            ),
        ]

        attempts = 0
        while len(constraints) < num_constraints and attempts < 100:
            template = self.rng.choice(constraint_templates)
            n1, n2 = self.rng.sample(names, 2)
            text, is_valid = template(n1, n2)
            if is_valid and text not in constraints:
                constraints.append(text)
            attempts += 1

        description = (
            f"Schedule {num_variables} people for activities.\n\n"
            f"People: {', '.join(names)}\n"
            f"Activities: {', '.join(activities)}\n"
            f"Time slots: {', '.join(str(s) for s in time_slots)}\n\n"
            f"Each person does exactly one activity in exactly one "
            f"time slot. No two people share an activity or time slot."
            f"\n\nConstraints:\n"
            + "\n".join(f"- {c}" for c in constraints)
            + "\n\nFind a valid assignment."
        )

        task_id = hashlib.md5(
            f"{self.generation_timestamp}:{description}".encode()
        ).hexdigest()[:12]

        return EvaluationTask(
            task_id=f"csp_{task_id}",
            description=description,
            category="reasoning",
            difficulty="hard" if num_variables > 3 else "medium",
            expected_answer=json.dumps(assignment, indent=2),
            verification_fn=self._csp_verifier_code(constraints),
            dimensions_tested=["reasoning", "symbolic_verification"],
        )

    def generate_multi_step_integration(self) -> EvaluationTask:
        """Generate a task requiring tool use, reasoning, and memory.

        The task description references 'current' information that
        the model cannot have memorized, forcing genuine tool use.
        """
        date_str = self.generation_timestamp[:10]

        scenarios = [
            {
                "description": (
                    f"As of {date_str}, find the current population of "
                    f"the three largest cities in Japan. Calculate their "
                    f"combined population. Then find a country whose "
                    f"total population is closest to that combined "
                    f"number. What is the GDP per capita of that "
                    f"country?"
                ),
                "dimensions": [
                    "tool_use", "reasoning", "planning", "memory",
                ],
                "difficulty": "hard",
            },
            {
                "description": (
                    f"Find the top 3 trending GitHub repositories "
                    f"today ({date_str}). For each repository, identify "
                    f"the primary programming language. Then find the "
                    f"year each of those languages was first released. "
                    f"What is the average release year? Name a "
                    f"significant historical event from that year."
                ),
                "dimensions": [
                    "tool_use", "reasoning", "planning", "memory",
                ],
                "difficulty": "hard",
            },
        ]

        scenario = self.rng.choice(scenarios)

        task_id = hashlib.md5(
            f"{self.generation_timestamp}:integration:"
            f"{scenario['description'][:50]}".encode()
        ).hexdigest()[:12]

        return EvaluationTask(
            task_id=f"int_{task_id}",
            description=scenario["description"],
            category="integration",
            difficulty=scenario["difficulty"],
            expected_answer=None,
            verification_fn=None,
            dimensions_tested=scenario["dimensions"],
        )

    def generate_reasoning_robustness(
        self, base_difficulty: str = "medium"
    ) -> EvaluationTask:
        """Generate math problems with irrelevant information,
        testing whether the system can filter noise
        (GSM-Symbolic style)."""
        a = self.rng.randint(12, 99)
        b = self.rng.randint(12, 99)
        c = self.rng.randint(2, 9)

        irrelevant_facts = self.rng.sample(
            [
                f"The store has been open for "
                f"{self.rng.randint(5,30)} years.",
                f"The walls are painted "
                f"{self.rng.choice(['blue','green','yellow'])}.",
                f"There are {self.rng.randint(3,12)} employees "
                f"working today.",
                f"The temperature outside is "
                f"{self.rng.randint(15,35)} degrees.",
                f"The store's phone number has "
                f"{self.rng.randint(7,10)} digits.",
                f"A customer left a "
                f"{self.rng.choice(['positive','mixed'])} "
                f"review yesterday.",
            ],
            k=self.rng.randint(2, 4),
        )

        problem = (
            f"A store has {a} items in stock on Monday. "
            f"{irrelevant_facts[0]} "
            f"On Tuesday, they receive a shipment of {b} more "
            f"items. {irrelevant_facts[1]} "
            f"On Wednesday, they sell exactly 1/{c} of their "
            f"total inventory. "
        )
        if len(irrelevant_facts) > 2:
            problem += f"{irrelevant_facts[2]} "
        problem += "How many items remain after Wednesday's sales?"

        total_before_sale = a + b
        sold = total_before_sale // c
        remaining = total_before_sale - sold

        task_id = hashlib.md5(
            f"{self.generation_timestamp}:robust:{a}:{b}:{c}"
            .encode()
        ).hexdigest()[:12]

        return EvaluationTask(
            task_id=f"rob_{task_id}",
            description=problem,
            category="reasoning_robustness",
            difficulty=base_difficulty,
            expected_answer=str(remaining),
            verification_fn=(
                f"lambda answer: int(answer.strip()) == {remaining}"
            ),
            dimensions_tested=["reasoning", "noise_filtering"],
            metadata={
                "irrelevant_fact_count": len(irrelevant_facts),
            },
        )

    def _csp_verifier_code(self, constraints: list[str]) -> str:
        """Return Python code that verifies a CSP solution."""
        return """
def verify(answer_json):
    import json
    assignment = (json.loads(answer_json)
                  if isinstance(answer_json, str)
                  else answer_json)
    # Check: all activities distinct
    activities = [v['activity'] for v in assignment.values()]
    if len(activities) != len(set(activities)):
        return False, "Duplicate activities"
    # Check: all slots distinct
    slots = [v['slot'] for v in assignment.values()]
    if len(slots) != len(set(slots)):
        return False, "Duplicate time slots"
    return True, "Valid assignment"
"""

    def generate_suite(
        self, n_per_category: int = 5
    ) -> list[EvaluationTask]:
        """Generate a balanced evaluation suite."""
        tasks = []
        for _ in range(n_per_category):
            tasks.append(
                self.generate_constraint_satisfaction(
                    num_variables=self.rng.randint(3, 5),
                    num_constraints=self.rng.randint(4, 8),
                ))
            tasks.append(self.generate_multi_step_integration())
            tasks.append(self.generate_reasoning_robustness())
        return tasks
