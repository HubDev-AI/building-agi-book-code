# prompt_evolver.py

class PromptEvolver:
    """Evolves system prompts and strategies
    based on what works."""

    def __init__(self, memory: 'SemanticMemory'):
        self.memory = memory
        self._strategy_wins: dict[str, int] = {}
        self._strategy_losses: dict[str, int] = {}

    def record_outcome(self, strategy: str,
                       task_type: str, success: bool):
        """Record whether a strategy worked for a
        task type."""
        key = f"{task_type}:{strategy}"
        if success:
            self._strategy_wins[key] = \
                self._strategy_wins.get(key, 0) + 1
        else:
            self._strategy_losses[key] = \
                self._strategy_losses.get(key, 0) + 1

    def get_best_strategy(self, task_type: str) -> str:
        """Return the strategy with the best win rate
        for this task type."""
        best_strategy = "reason_then_verify"  # default
        best_rate = 0.0

        for key, wins in self._strategy_wins.items():
            if key.startswith(f"{task_type}:"):
                strategy = key.split(":", 1)[1]
                losses = self._strategy_losses.get(key, 0)
                total = wins + losses
                if total >= 3:  # Minimum sample
                    rate = wins / total
                    if rate > best_rate:
                        best_rate = rate
                        best_strategy = strategy
        return best_strategy

    async def evolve_prompt(
        self, base_prompt: str,
        task_type: str, feedback: str,
    ) -> str:
        """Evolve a system prompt based on accumulated
        feedback.

        This is the lightest form of continual learning:
        the model's weights don't change, but its
        instructions do.
        """
        winning_strategies = []
        for key, wins in self._strategy_wins.items():
            if key.startswith(f"{task_type}:"):
                losses = self._strategy_losses.get(key, 0)
                rate = wins / (wins + losses)
                if rate > 0.7:
                    winning_strategies.append(
                        key.split(":", 1)[1]
                    )

        evolved = base_prompt
        if winning_strategies:
            evolved += (
                f"\n\nBased on experience, these "
                f"strategies work well for {task_type} "
                f"tasks: {', '.join(winning_strategies)}. "
                f"Prefer these approaches."
            )
        if feedback:
            evolved += (
                f"\n\nRecent feedback to incorporate: "
                f"{feedback}"
            )

        return evolved
